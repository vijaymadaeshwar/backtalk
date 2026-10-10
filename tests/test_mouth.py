"""mouth.py offline: the sentence logic, the engine fallback chain, the
playback queue, and the credential lookup -- none of it touching a device.

Kokoro, sounddevice and espeak-ng are all faked or stubbed, so this runs
headless in CI. What it pins is the behaviour that is easy to break and
hard to notice: which engine speaks when the premium one fails, that the
one long-lived output stream is reused, that a barge-in actually cuts the
audio, and that the ElevenLabs key never comes from a file.
"""
from pathlib import Path
import os
import shutil
import sys
import tempfile
import time
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import backtalk.signals as sig  # noqa: E402
from backtalk import mouth  # noqa: E402
from backtalk.mouth import Mouth  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'ok' if cond else 'FAIL'} {name}"
          + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILURES.append(name)


class FakeOut:
    def __init__(self, active=True):
        self.active = active
        self.writes: list = []
        self.started = 0
        self.closed = False

    def start(self):
        self.started += 1

    def write(self, data):
        self.writes.append(data)

    def close(self, ignore_errors=False):
        self.closed = True


class DeadOut:
    """A stream whose .active raises, the way a device gone away does."""
    def __init__(self):
        self.started = 0
        self.closed = False

    @property
    def active(self):
        raise RuntimeError("device gone")

    def start(self):
        self.started += 1

    def write(self, data):
        pass

    def close(self, ignore_errors=False):
        self.closed = True


def quiet_mouth() -> Mouth:
    """A Mouth with no worker loop running, so tests drive the queue."""
    with mock.patch.object(Mouth, "_run", lambda self: None):
        return Mouth()


def sentences():
    print("\n--- split_sentences ---")
    check("splits on punctuation",
          mouth.split_sentences("One. Two! Three?") ==
          ["One.", "Two!", "Three?"], mouth.split_sentences("One. Two!"))
    check("collapses extra whitespace",
          mouth.split_sentences("One.   Two") == ["One.", "Two"])
    check("no punctuation -> one part",
          mouth.split_sentences("just a line") == ["just a line"])
    check("blank -> empty list", mouth.split_sentences("   ") == [])


def voices():
    print("\n--- voice_for / _voices ---")
    with mock.patch.object(mouth, "CFG",
                           {"voices": {"en": "voiceEN", "a": "voiceA"}}):
        check("table entry returned", mouth.voice_for("a") == "voiceA")
        check("unknown language falls back to the table's en",
              mouth.voice_for("zz") == "voiceEN", mouth.voice_for("zz"))
    with mock.patch.object(mouth, "CFG", {"voice": "myvoice"}):
        check("no voices table -> single default",
              mouth._voices() == {"en": "myvoice"}, mouth._voices())
        check("matching language uses the default",
              mouth.voice_for("en") == "myvoice")


def espeak_choices():
    print("\n--- espeak_voice_for ---")
    with mock.patch.object(mouth, "_ESPEAK", "/usr/bin/espeak-ng"):
        check("zh -> cmn (Mandarin, not a region)",
              mouth.espeak_voice_for("zh") == "cmn")
        check("yue -> yue even though it is three letters",
              mouth.espeak_voice_for("yue") == "yue")
        check("nb -> nn", mouth.espeak_voice_for("nb") == "nn")
        check("region stripped: en-US -> en",
              mouth.espeak_voice_for("en-US") == "en")
        check("underscore variant normalised",
              mouth.espeak_voice_for("pt_BR") == "pt")
        check("bare three-letter code -> None (let espeak guess)",
              mouth.espeak_voice_for("tam") is None)
        check("None -> None", mouth.espeak_voice_for(None) is None)
    with mock.patch.object(mouth, "_ESPEAK", ""):
        check("no espeak binary -> None", mouth.espeak_voice_for("en") is None)


def espeak_early_returns():
    print("\n--- _stream_espeak early exits (no binary needed) ---")
    with mock.patch.object(mouth, "_ESPEAK", ""):
        check("no binary -> no audio",
              list(mouth._stream_espeak("hi", "en")) == [])
    with mock.patch.object(mouth, "_ESPEAK", "/x/espeak"), \
         mock.patch.object(mouth, "espeak_voice_for", return_value=None):
        check("language espeak cannot speak -> silence",
              list(mouth._stream_espeak("hi", "xx")) == [])
    with mock.patch.object(mouth, "_ESPEAK", "/x/espeak"), \
         mock.patch.object(mouth, "espeak_voice_for", return_value="en"):
        check("blank text -> silence",
              list(mouth._stream_espeak("   ", "en")) == [])


def synth_chain():
    print("\n--- synth_stream: the engine fallback chain ---")
    pcm = np.zeros(4, dtype=np.int16)
    with mock.patch.object(mouth, "log"):
        with mock.patch.object(mouth, "_elevenlabs_ready", return_value=True), \
             mock.patch.object(mouth, "_stream_elevenlabs",
                               side_effect=lambda *a, **k: iter([pcm])):
            out = list(mouth.synth_stream("hi"))
        check("elevenlabs used when ready and keyed",
              bool(out) and out[0][0] == mouth.EL_RATE, out)

        with mock.patch.object(mouth, "_elevenlabs_ready", return_value=True), \
             mock.patch.object(mouth, "_stream_elevenlabs",
                               side_effect=RuntimeError("cloud down")), \
             mock.patch.object(mouth, "_stream_kokoro",
                               side_effect=lambda t: iter([pcm])):
            out = list(mouth.synth_stream("hi"))
        check("elevenlabs failure degrades to kokoro",
              bool(out) and out[0][0] == mouth.KOKORO_RATE, out)

        with mock.patch.object(mouth, "_elevenlabs_ready", return_value=False), \
             mock.patch.object(mouth, "_stream_kokoro",
                               side_effect=lambda t: iter([])), \
             mock.patch.object(mouth, "_stream_espeak",
                               side_effect=lambda t, l: iter([pcm])), \
             mock.patch.object(mouth, "_espeak_rate", 22050):
            out = list(mouth.synth_stream("hi"))
        check("empty kokoro falls through to espeak",
              bool(out) and out[0][0] == 22050, out)

        with mock.patch.object(mouth, "_elevenlabs_ready", return_value=False), \
             mock.patch.object(mouth, "_stream_kokoro",
                               side_effect=RuntimeError("no pipe")), \
             mock.patch.object(mouth, "_stream_espeak",
                               side_effect=RuntimeError("no espeak")):
            out = list(mouth.synth_stream("hi"))
        check("every engine failing -> no audio at all", out == [], out)


def keys():
    print("\n--- the ElevenLabs key comes from a store, not a file ---")
    cfg = {"elevenlabs": {"key_slot": "custom-slot"}}
    with mock.patch.object(mouth, "CFG", cfg):
        check("key slot honours config",
              mouth._key_slot() == "custom-slot", mouth._key_slot())
    with mock.patch.object(mouth, "CFG", {"elevenlabs": {}}):
        check("key slot default", mouth._key_slot() == "backtalk-elevenlabs")

    mouth._el_key_cache = None
    run = mock.Mock(return_value=mock.Mock(returncode=0, stdout="secret\n"))
    with mock.patch.object(mouth, "CFG", cfg), \
         mock.patch.object(mouth.sys, "platform", "darwin"), \
         mock.patch.object(mouth.subprocess, "run", run):
        got = mouth._get_elevenlabs_key()
    check("macOS keychain path returns the secret", got == "secret", got)
    with mock.patch.object(mouth.subprocess, "run") as run2:
        again = mouth._get_elevenlabs_key()
    check("cached: the store is not read twice",
          again == "secret" and not run2.called, again)

    mouth._el_key_cache = None
    with mock.patch.object(mouth, "CFG", cfg), \
         mock.patch.object(mouth.sys, "platform", "linux"), \
         mock.patch.object(mouth.subprocess, "run",
                           return_value=mock.Mock(returncode=1, stdout="")):
        os.environ["ELEVENLABS_API_KEY"] = "envkey"
        try:
            got = mouth._get_elevenlabs_key()
        finally:
            os.environ.pop("ELEVENLABS_API_KEY", None)
    check("failed store lookup falls back to the env var",
          got == "envkey", got)

    mouth._el_key_cache = None
    with mock.patch.object(
            mouth, "CFG",
            {"elevenlabs": {"enabled": True, "voice_id": "v", "model": "m"}}), \
         mock.patch.object(mouth, "_get_elevenlabs_key", return_value="k"):
        check("ready with enabled+voice id+key", mouth._elevenlabs_ready())
    with mock.patch.object(mouth, "CFG",
                           {"elevenlabs": {"enabled": True, "model": "m"}}), \
         mock.patch.object(mouth, "_get_elevenlabs_key", return_value="k"):
        check("no voice id -> not ready", not mouth._elevenlabs_ready())
    with mock.patch.object(
            mouth, "CFG",
            {"elevenlabs": {"enabled": True, "voice_id": "v", "model": "m"}}), \
         mock.patch.object(mouth, "_get_elevenlabs_key", return_value=""):
        check("no key -> not ready", not mouth._elevenlabs_ready())
    with mock.patch.object(
            mouth, "CFG",
            {"elevenlabs": {"voice_id": "v", "model": "m"}}), \
         mock.patch.object(mouth, "_get_elevenlabs_key", return_value="k"):
        check("disabled -> not ready", not mouth._elevenlabs_ready())


def sweep():
    print("\n--- orphan espeak scratch dirs ---")
    root = tempfile.mkdtemp(prefix="bt-test-")
    try:
        orphan = os.path.join(root, "orphan")
        os.mkdir(orphan)
        open(os.path.join(orphan, "espeak-ng.dll"), "w").close()
        used = os.path.join(root, "used")
        os.mkdir(used)
        open(os.path.join(used, "espeak-ng.dll"), "w").close()
        open(os.path.join(used, "other.txt"), "w").close()
        empty = os.path.join(root, "empty")
        os.mkdir(empty)
        plain = os.path.join(root, "note.txt")
        open(plain, "w").close()
        check("one-library dir is an orphan",
              mouth._is_orphan_espeak_tempdir(orphan))
        check("two-file dir is not",
              not mouth._is_orphan_espeak_tempdir(used))
        check("empty dir is not", not mouth._is_orphan_espeak_tempdir(empty))
        check("missing path is not",
              not mouth._is_orphan_espeak_tempdir(
                  os.path.join(root, "nope")))
        with mock.patch.object(mouth.tempfile, "gettempdir",
                               return_value=root), \
             mock.patch.object(mouth, "log"):
            mouth._sweep_orphan_espeak_tempdirs()
        check("orphan removed", not os.path.exists(orphan))
        check("dir with another file kept", os.path.exists(used))
        check("plain file kept", os.path.exists(plain))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def queue_behaviour():
    print("\n--- the Mouth queue (no worker) ---")
    m = quiet_mouth()
    m.say("One. Two. Three.")
    got = []
    while not m._q.empty():
        got.append(m._q.get_nowait())
    check("say splits into sentences",
          [g[0] for g in got] == ["One.", "Two.", "Three."], got)

    m.say_chunk("  a single chunk  ", ["wave"])
    item = m._q.get_nowait()
    check("say_chunk keeps one request and strips it",
          item == ("a single chunk", ["wave"]), item)
    m.say_chunk("   ")
    check("blank say_chunk queues nothing", m._q.empty())

    m.say("One. Two.")
    m.shut_up()
    check("shut_up flushes the queue", m._q.empty())
    check("shut_up raises the stop flag", m._stop.is_set())

    m2 = quiet_mouth()
    check("speaking reflects the event", not m2.speaking)
    m2._speaking.set()
    check("speaking set", m2.speaking)

    m3 = quiet_mouth()
    t0 = time.time()
    m3.wait_done(timeout=0.2)
    check("wait_done returns at once on an empty queue",
          time.time() - t0 < 0.15, time.time() - t0)

    m4 = quiet_mouth()
    m4.ducker = mock.Mock()
    m4.shutdown()
    check("shutdown restores the music immediately",
          m4.ducker.restore_now.called)


def worker_loop():
    print("\n--- the worker loop parks the bus when speech drains ---")
    played = []
    with mock.patch.object(Mouth, "_play_stream",
                           new=lambda self, s, d=None: played.append((s, d))), \
         mock.patch.object(sig, "static_stop"), \
         mock.patch.object(sig, "set_state") as set_state, \
         mock.patch.object(sig, "caption"), \
         mock.patch.object(sig, "caption_clear") as cap_clear, \
         mock.patch.object(sig, "reply_done") as reply_done, \
         mock.patch.object(sig, "feed_waveform"), \
         mock.patch.object(sig, "direction"):
        m = Mouth()
        m.say_chunk("Hello there.", ["wave"])
        deadline = time.time() + 5
        while not reply_done.called and time.time() < deadline:
            time.sleep(0.01)
        m.shut_up()
    check("the sentence was handed to playback",
          played == [("Hello there.", ["wave"])], played)
    check("speaking was announced", set_state.called)
    states = [c.args[0] for c in set_state.call_args_list]
    check("bus goes speaking then idle",
          states[:1] == ["speaking"] and states[-1:] == ["idle"], states)
    check("caption cleared and reply marked done",
          cap_clear.called and reply_done.called)


def streams():
    print("\n--- the one long-lived output stream ---")
    m = quiet_mouth()
    made = []

    def new_stream(**k):
        s = FakeOut()
        made.append(s)
        return s

    with mock.patch.object(mouth.sd, "OutputStream", side_effect=new_stream):
        a = m._get_out(24000)
        b = m._get_out(24000)
        c = m._get_out(44100)
    check("reused while the rate is unchanged", a is b)
    check("reopened when the rate changes", c is not a)
    check("exactly two streams built for two rates", len(made) == 2, len(made))

    m2 = quiet_mouth()
    inactive = FakeOut(active=False)
    with mock.patch.object(mouth.sd, "OutputStream",
                           side_effect=lambda **k: inactive):
        x = m2._get_out(24000)
        y = m2._get_out(24000)
    check("an inactive stream is restarted, not rebuilt",
          x is y and inactive.started >= 2, inactive.started)

    m3 = quiet_mouth()
    dead, fresh = DeadOut(), FakeOut()
    seq = iter([dead, fresh])
    with mock.patch.object(mouth.sd, "OutputStream",
                           side_effect=lambda **k: next(seq)), \
         mock.patch.object(mouth, "log"):
        first = m3._get_out(24000)
        second = m3._get_out(24000)
    check("a stream that raises .active is dropped and replaced",
          first is dead and second is fresh)
    check("the dead stream is closed on the way out", dead.closed)


def teardown_paths():
    print("\n--- _cut / _drop_out ---")
    m = quiet_mouth()
    m._out = FakeOut()
    m._cut()
    check("cut writes three beats of silence",
          len(m._out.writes) == 3, len(m._out.writes))
    quiet_mouth()._cut()  # no stream: must not raise
    check("cut with no stream is safe", True)

    m2 = quiet_mouth()
    fake = FakeOut()
    m2._out, m2._out_rate = fake, 24000
    m2._drop_out()
    check("drop closes and forgets the stream",
          fake.closed and m2._out is None and m2._out_rate is None)


def playback():
    print("\n--- _play_stream: buffer, write, publish, cut ---")

    def one_chunk(text, timeout=30.0):
        yield (24000, np.zeros(24000, dtype=np.int16))

    m = quiet_mouth()
    fake = FakeOut()
    with mock.patch.object(mouth, "synth_stream", side_effect=one_chunk), \
         mock.patch.object(Mouth, "_get_out", return_value=fake), \
         mock.patch.object(sig, "feed_waveform") as feed, \
         mock.patch.object(sig, "direction") as direction, \
         mock.patch.object(sig, "set_state"), \
         mock.patch.object(sig, "caption"), \
         mock.patch.object(sig, "caption_clear"), \
         mock.patch.object(sig, "reply_done"):
        m._play_stream("Hello there.", ["wave"])
    check("audio was written to the stream", len(fake.writes) > 0)
    check("waveform fed to the bus", feed.called)
    check("directions published when audio starts",
          direction.called and direction.call_args.args[0] == ["wave"])

    m2 = quiet_mouth()
    m2._stop.set()
    cut = []
    with mock.patch.object(mouth, "synth_stream", side_effect=one_chunk), \
         mock.patch.object(Mouth, "_get_out", return_value=FakeOut()), \
         mock.patch.object(Mouth, "_cut",
                           lambda self: cut.append(True)), \
         mock.patch.object(sig, "feed_waveform"), \
         mock.patch.object(sig, "direction"), \
         mock.patch.object(sig, "set_state"), \
         mock.patch.object(sig, "caption"), \
         mock.patch.object(sig, "caption_clear"), \
         mock.patch.object(sig, "reply_done"):
        m2._play_stream("Hello there.")
    check("a raised stop flag cuts before any audio", bool(cut) and True)

    m3 = quiet_mouth()
    with mock.patch.object(mouth, "synth_stream",
                           side_effect=lambda t, timeout=30.0: iter([])):
        m3._play_stream("nothing at all")
    check("no synthesized audio -> returns quietly", m3._out is None)


print("=" * 66)
print("THE MOUTH (offline, no device, no model)")
print("=" * 66)
sentences()
voices()
espeak_choices()
espeak_early_returns()
synth_chain()
keys()
sweep()
queue_behaviour()
worker_loop()
streams()
teardown_paths()
playback()

print("\n" + "=" * 66)
print("MOUTH OK" if not FAILURES else f"MOUTH FAILURES: {FAILURES}")
sys.exit(1 if FAILURES else 0)
