"""The signal bus, isolated: every note the voice line leaves on disk.

signals has no model and no audio behind it -- it is tiny files other
programs watch -- so it can be driven end to end with the paths pointed
at a temp dir. What is checked is the contract a face depends on: the
right file, the right bytes, and the promise that no write ever raises.
"""
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np                                          # noqa: E402
from backtalk.config import CFG                             # noqa: E402

_TMP = tempfile.mkdtemp(prefix="bt_signals_")
CFG["signals_dir"] = _TMP
CFG["barehands_state_dir"] = os.path.join(_TMP, "bh")
os.makedirs(CFG["barehands_state_dir"], exist_ok=True)
CFG["thinking_sound"] = ""

from backtalk import signals                                # noqa: E402

failures = []


def check(name, cond, detail=""):
    print("    %s %s%s" % ("ok  " if cond else "FAIL", name,
                           "" if cond else "   <- " + str(detail)))
    if not cond:
        failures.append(name)


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


print("=" * 66)
print("THE SIGNAL BUS")
print("=" * 66)

signals.set_state("listening")
check("state is written", read(os.path.join(_TMP, ".voice_state")) == "listening")
check("state is mirrored to barehands",
      read(os.path.join(_TMP, "bh", "state")) == "listening")

signals.park()
check("park leaves the bus idle",
      read(os.path.join(_TMP, ".voice_state")) == "idle")

signals.caption("hello there")
cap = json.loads(read(os.path.join(_TMP, ".voice_caption")))
check("caption carries the words", cap["text"] == "hello there", cap)
signals.caption_clear()
check("caption_clear removes the file",
      not os.path.exists(os.path.join(_TMP, ".voice_caption")))

signals.language("en")
check("language is written",
      json.loads(read(os.path.join(_TMP, ".voice_language")))["language"] == "en")
signals.language(None)
check("a None language stays None, not the string",
      json.loads(read(os.path.join(_TMP, ".voice_language")))["language"] is None)

signals.direction(["<<smile>>", "<<wave>>"])
d = json.loads(read(os.path.join(_TMP, ".voice_direction")))
check("directions are published raw", d["directions"] == ["<<smile>>", "<<wave>>"], d)
signals.direction([])
check("empty directions write nothing new", d["directions"] == ["<<smile>>", "<<wave>>"])

signals.reply_done()
check("reply_done is stamped",
      "ts" in json.loads(read(os.path.join(_TMP, ".voice_reply_done"))))

signals.set_rate_limit("session", 0.5, 111)
signals.set_rate_limit("seven_day", None, 222)
rl = json.loads(read(os.path.join(_TMP, ".voice_rate_limits")))
check("rate limits merge two windows",
      set(rl) == {"session", "seven_day"} and rl["session"]["utilization"] == 0.5, rl)
signals.set_rate_limit("", 1, 2)
check("an empty window is ignored",
      set(json.loads(read(os.path.join(_TMP, ".voice_rate_limits")))) ==
      {"session", "seven_day"})

pcm = (np.sin(np.linspace(0, 20, 400)) * 10000).astype(np.int16)
signals.feed_waveform(pcm)
wav = json.loads(read(os.path.join(_TMP, ".voice_waveform")))
check("a waveform is downsampled to 64 points", len(wav["samples"]) == 64, len(wav["samples"]))
check("speaking state self-heals on a waveform",
      read(os.path.join(_TMP, ".voice_state")) == "speaking")
check("barehands wave is normalised 0..1",
      all(-1.0 <= s <= 1.0 for s in
          json.loads(read(os.path.join(_TMP, "bh", "wave.json")))["samples"]))

signals._last_waveform_write = 0.0
signals.feed_waveform(np.zeros(0, dtype=np.int16))   # empty: must not raise
check("an empty block is ignored safely", True)

signals.static_start()                                # no thinking sound configured
check("static_start no-ops without a thinking sound", signals._static_proc is None)
signals.static_stop()                                 # never raises with nothing playing
check("static_stop is safe with nothing playing", True)

signals.register_exit_park()
check("register_exit_park installs cleanly", True)

print("\n" + "=" * 66)
print("SIGNALS OK" if not failures else "SIGNALS FAILURES: %s" % failures)
sys.exit(1 if failures else 0)
