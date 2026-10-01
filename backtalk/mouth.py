# backtalk: talk to your opencode agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The mouth — streaming sentence-chunked TTS, played through one
long-lived output stream.

Default engine: Kokoro, in-process. Local, free, no server, no API key,
~0.2s to first audio once warm. Optional premium engine: ElevenLabs on
YOUR key — read from the system keychain, never from a file (see
_get_elevenlabs_key) — with Kokoro as the automatic fallback: the voice
degrades instead of going mute if the cloud fails.

Sentences are synthesized one at a time and queued for playback, so the
first sentence is audible while later ones are still rendering. Playback
is cancellable mid-word: set the stop event and the speaker goes silent
within one audio block plus the device buffer (~0.15s).

HARD-WON AUDIO LAW #1 — ONE long-lived OutputStream, reused for every
sentence for the life of the process. A fresh stream per sentence gives
an audible onset blip or a beat of dead air on plenty of audio setups
(USB interfaces, Bluetooth, streaming mixers that latch onto each new
stream late). Proven by A/B test; do not "simplify" this away.

HARD-WON AUDIO LAW #2 — buffer ~0.75s of synthesized audio before a
sentence starts playing, so a slower machine never underruns into
slow-motion garble.
"""
import io
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading

import numpy as np
import sounddevice as sd

from backtalk.config import CFG
from backtalk.vlog import log

KOKORO_RATE = 24000
EL_RATE = 44100
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

_pipe = None
_pipes: dict = {}  # kokoro pipeline per language letter, loaded on demand
_pipe_lock = threading.Lock()


def _ensure_espeak():
    """kokoro phonemizes through system espeak-ng (its bundled loader
    ships a broken build path — found the hard way; upstream's own docs
    say install the system package). Help phonemizer find it in the
    usual homes when the env isn't already set."""
    if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        return
    candidates = (
        "/opt/homebrew/lib/libespeak-ng.dylib",       # macOS arm64 (brew)
        "/usr/local/lib/libespeak-ng.dylib",          # macOS intel (brew)
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",  # debian/ubuntu
        "/usr/lib/libespeak-ng.so.1",                 # other linux
        "C:\\Program Files\\eSpeak NG\\libespeak-ng.dll",       # windows
        "C:\\Program Files (x86)\\eSpeak NG\\libespeak-ng.dll",
    )
    for lib in candidates:
        if os.path.exists(lib):
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = lib
            break


# Every espeak library filename phonemizer might copy, on any platform. A
# directory holding exactly one of these and nothing else is a phonemizer
# scratch dir and is not plausibly anything else.
_ESPEAK_LIB_NAMES = (
    "espeak-ng.dll",
    "libespeak-ng.dll",
    "libespeak-ng.so",
    "libespeak-ng.so.1",
    "libespeak-ng.dylib",
)


def _is_orphan_espeak_tempdir(path: str) -> bool:
    """True only for a directory whose ENTIRE contents are one espeak
    library. That signature is what makes it safe to point a delete at a
    shared temp folder: one file, and its name is one of five."""
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    return len(entries) == 1 and entries[0] in _ESPEAK_LIB_NAMES


def _sweep_orphan_espeak_tempdirs():
    """Delete espeak scratch dirs left behind by previous runs.

    phonemizer copies the espeak shared library into a fresh temp dir for
    every backend it builds, because espeak-ng keeps its state in globals
    and the loader refuses the same file twice. Kokoro builds several
    backends, so ONE start leaves several behind.

    On POSIX that cleanup rides a finalizer and usually happens. On
    Windows phonemizer can only register it with atexit, and atexit does
    not run when a process is KILLED rather than exited -- so anything
    stopping the voice line by terminating it, which is most launchers and
    every supervisor, leaks every directory it ever made. Sixty had piled
    up on the machine where this was found, and fifteen were sitting on
    the author's own Mac when it was reviewed: the POSIX path is not as
    reliable as it looks either. The count only ever grows.

    Patching phonemizer where it is installed is not a fix, because the
    launcher runs a dependency sync that would overwrite it. Sweeping at
    our own startup bounds the total at one run's worth instead.

    Two things make deleting from a shared temp folder safe, and only the
    first is ours: the signature above is narrow enough that nothing else
    matches it, and anything we are not permitted to remove raises and is
    skipped. On Windows a loaded library cannot be deleted at all, so a
    live instance is protected by the OS rather than by us noticing it.
    POSIX does not work that way, but a process that has already mapped
    the library keeps it after the unlink, so a running instance is
    unharmed either way.
    """
    root = tempfile.gettempdir()
    swept = 0
    try:
        names = os.listdir(root)
    except OSError:
        return
    for name in names:
        path = os.path.join(root, name)
        if not os.path.isdir(path) or not _is_orphan_espeak_tempdir(path):
            continue
        try:
            shutil.rmtree(path)
            swept += 1
        except OSError:
            pass          # in use, or not ours. Leaving it is correct.
    if swept:
        log(f"[mouth] swept {swept} orphaned espeak temp dir(s)")


# Whisper's language code -> Kokoro voice, from CFG["voices"]. English is
# the default for anything unmapped or undetectable.
_LANG_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("zh", ("chinese", "mandarin", "cantonese", "putonghua")),
    ("ja", ("japanese", "tokyo")),
    ("hi", ("hindi", "devanagari")),
    ("es", ("spanish", "castellano", "espanol")),
    ("pt", ("portuguese", "portugues", "brazilian")),
    ("fr", ("french", "francais")),
    ("it", ("italian", "italiano")),
    ("de", ("german", "deutsch")),
)

# Words so characteristic of one language that finding even one settles it.
# Needed because romanised speech ("ni hao", "konnichiwa", "buongiorno") has
# no accents and almost no overlap with the stopword lists, so the score
# below would call those Portuguese or Hindi at random.
_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("zh", ("ni hao", "hao ma", "zhen de", "zen me", "shi shen",
            "qing zhu", "bu ke yi", "zai na", "zhe ge", "zhong guo")),
    ("ja", ("konnichiwa", "genki", "arigato", "ohayou", "sayounara",
            "kudasai", "imasu", "desu ka", "hajimemashite")),
    ("hi", ("namaste", "kaise", "kaisa", "dhanyavad", "aap kaise",
            "kya kar", "theek hai", "nahi hai", "main aap")),
    ("it", ("buongiorno", "buonasera", "come stai", "per favore",
            "non posso", "aprire il", "il file", "grazie")),
    # No bare "ola" here: that word is spelled the same in Portuguese and
    # Spanish, so it decides nothing. Portuguese is identified by its
    # longer forms instead, and the shared word goes to Spanish, where it
    # is more common on its own.
    ("pt", ("bom dia", "boa noite", "esta bem", "tudo bem", "obrigado",
            "vou abrir", "o arquivo", "como voce", "nao posso")),
    ("es", ("hola", "buenos dias", "buenas", "por favor", "gracias",
            "voy a abrir", "el archivo", "como estas", "muy bien")),
    ("fr", ("bonjour", "bonsoir", "s il vous plait", "merci",
            "je vais", "le fichier", "comment allez", "tres bien")),
    ("de", ("guten tag", "guten morgen", "bitte", "danke", "ich werde",
            "die datei", "wie geht", "sehr gut", "ich offne")),
)

# The most frequent words in each language. A spoken reply is built almost
# entirely from these, so scoring them identifies the language even when the
# text is written without accents or is only a few words long -- which is
# exactly the case for a one line answer. This is a stopword count, not
# real linguistics: it only has to pick between nine voices, and when it
# genuinely cannot tell, English is the safe answer.
_STOPWORDS: dict[str, frozenset] = {
    "es": frozenset("""de la que el en y a los se del las un por con no una su
        para es al lo como mas pero sus le ya o este si porque esta entre
        cuando muy sin sobre tambien me hasta hay donde quien desde todo nos
        durante ti han yo hay vez puede estan""".split()),
    "fr": frozenset("""le de un etre et a il avoir ne je son que se qui ce dans
        en du elle au pour que pas vous par sur faire plus dire me on mon il ne
        nous comme mais ou si les leur tout bien ete etre a moi toi son tes
        avec ce il qui nous vous ils cette est""".split()),
    "de": frozenset("""der die und in den von zu das mit sich des auf fur ist im
        dem nicht ein eine als auch es an werden aus er hat dass sie nach wird
        bei einer um am sind noch wie einem uber einen so zum haben nur oder
        aber vor zur bis mehr durch man sein wurde sei""".split()),
    "it": frozenset("""di che e il la un per in una sono con non si da come ma le
        lo ci questo al del dei della nel alla anche gli suo piu o ma se mi
        ho ha te ne cosa quando molto dove chi perche tutto io essere fare
        della degli""".split()),
    "pt": frozenset("""de que nao uma dos como mas foi ao ele das tem a e os do
        da no por mais as dos como mas ao ele das tem um para com uma nao
        voce ja esta eu muito quando onde porque""".split()),
    "hi": frozenset("""hai aap hai kaha kya kar rahe hain na hi main wo ye ki ka
        se hai par kyun kaise kab kahan kuch bahut accha""".split()),
    "ja": frozenset("""の に は を た が で て と し れ さ ある いる も する から
        な こと として い や れる など なっ ない この ため その あっ よう また
        こと これ する んだ 私 ので す""".split()),
    "zh": frozenset("""的 了 是 我 你 他 她 我们 你们 这 那 在 有 和 就 不 人 都
        一 一个 上 也 很 到 说 要 去 会 着 没有 看 好 自己 这 那""".split()),
    "en": frozenset("""the of and to a in is it you that he was for on are as with
        his they i at be this have from or one had by word but not what all
        were we when your can said there use an each which she do how their
        if will up other about out many then them these so some her would make
        like him into time has look two more write go see number no way
        could people my than first water been call who oil its now find long
        down day did get come made may part""".split()),
}


def detect_language(text: str, hint: str | None = None) -> str:
    """Which language a reply is written in.

    Whisper's code wins when the caller already knows it (it heard the
    user). Otherwise we sniff the reply itself, because a Hindi question
    that gets an English answer must still be SPOKEN in English -- the
    text the mouth has is the only language that matters for the voice.
    Non-Latin scripts are the reliable signal; Latin-script languages are
    told apart by their own words, and anything ambiguous stays English
    rather than being read aloud in the wrong accent.
    """
    if hint:
        hint = str(hint).strip().lower()[:2]
        if hint in _voices():
            return hint
        # No kokoro voice for it, but espeak-ng can still speak it natively,
        # so the hint stands -- voice_for() maps it to an English voice and
        # synth_stream routes around that.
        if espeak_voice_for(hint):
            return hint
    low = (text or "").lower()
    if not low.strip():
        return "en"
    import re as _re
    # Kana is checked BEFORE Han on purpose: Japanese is written with Han
    # too, so a naive "any Han character means Chinese" test reads every
    # Japanese sentence as Chinese. Kana present at all settles it.
    #
    # The rest are scripts Kokoro has no voice for. They are detected so the
    # mouth can reach for espeak-ng instead of reading them in an English
    # accent -- but they are deliberately NOT trusted as a hint above,
    # because "en" there means "espeak will handle it", not "the voice
    # table has it".
    scripts = (
        (r"[\u3040-\u30ff]", "ja"),
        (r"[\u4e00-\u9fff]", "zh"),
        (r"[\u0900-\u097f]", "hi"),
        (r"[\u0b80-\u0bff]", "ta"),
        (r"[\uac00-\ud7af\u1100-\u11ff]", "ko"),
        (r"[\u0600-\u06ff]", "ar"),
        (r"[\u0400-\u04ff]", "ru"),
        (r"[\u0e00-\u0e7f]", "th"),
        (r"[\u0370-\u03ff]", "el"),
        (r"[\u0590-\u05ff]", "he"),
        (r"[\u0d00-\u0d7f]", "ml"),
        (r"[\u0c00-\u0c7f]", "kn"),
        (r"[\u0c80-\u0cff]", "gu"),
        (r"[\u0a00-\u0a7f]", "pa"),
        (r"[\u0980-\u09ff]", "bn"),
    )
    for pattern, code in scripts:
        if _re.search(pattern, text):
            return code
    for code, words in _LANG_HINTS:
        for w in words:
            # Require a word boundary in the raw text for the accented
            # spellings, but allow the bare ASCII forms anywhere.
            if _re.search(r"\b%s\b" % _re.escape(w), low):
                return code
    # Characteristic words, checked before the score. Ordered so a longer,
    # more specific phrase beats a shorter shared one.
    for code, phrases in _MARKERS:
        for p in phrases:
            if p in low:
                return code
    # Nothing distinctive jumped out, so score the everyday words of each
    # language against the reply. Strip accents first: a reply typed or
    # synthesised as "como estas" must still read as Spanish, not English.
    words = _strip_accents(low).split()
    if not words:
        return "en"
    best, best_score = "en", 0
    for code, table in _STOPWORDS.items():
        hits = sum(1 for w in words if w in table)
        # English is the fallback, so it only wins on an equal score.
        if hits > best_score or (hits == best_score > 0 and code == "en"):
            best, best_score = code, hits
    return best if best_score else "en"


_turn_hint = None  # language of the CURRENT turn, from the ear


def set_turn_language(lang: str | None) -> None:
    """Record the language the user just spoke, so a reply that echoes it
    is spoken in kind. Cleared by the next turn's transcribe."""
    global _turn_hint
    _turn_hint = (lang or None)


def _turn_lang() -> str | None:
    return _turn_hint


def _strip_accents(s: str) -> str:
    """Fold accents and punctuation off so word matching sees plain ASCII."""
    import unicodedata
    out = []
    for ch in s:
        if unicodedata.category(ch).startswith("P"):
            out.append(" ")
            continue
        out.append(ch)
    folded = unicodedata.normalize("NFKD", "".join(out))
    return "".join(c for c in folded if not unicodedata.combining(c))


def _voices() -> dict:
    v = CFG.get("voices") or {}
    return v if isinstance(v, dict) and v else {"en": CFG.get("voice") or "bm_lewis"}


# The languages Kokoro can actually speak in its own voice. Derived from
# what the model ships with, NOT from the voices table: that table also
# carries read-in-English entries for languages Kokoro has no voice for,
# so testing membership against it says "German is fine" when it is not.
KOKORO_LANGS = ("en", "es", "fr", "hi", "it", "ja", "pt", "zh")


def kokoro_langs() -> tuple:
    """Languages Kokoro renders natively. Anything else goes to espeak-ng
    so it is spoken in its own language instead of an English accent."""
    return KOKORO_LANGS


def voice_for(lang: str | None) -> str:
    """The voice for a language, falling back to the English default.

    A language with no configured voice is read in English rather than in
    some other language's accent -- Kokoro has no multilingual voice, so a
    mismatch would mangle the words instead of just sounding foreign.
    """
    table = _voices()
    if lang and lang in table:
        return table[lang]
    return table.get("en") or CFG.get("voice") or "bm_lewis"


# --- espeak-ng fallback -------------------------------------------------
# Kokoro ships 54 voices across 9 languages. Everything else -- Tamil,
# Korean, Arabic, Russian, Thai -- had no voice at all and was read in a
# British English accent, which is worse than useless for someone who can
# read the script but cannot make sense of the spoken words.
#
# espeak-ng covers ~100 languages including every one of those. It sounds
# robotic next to Kokoro, so it is a fallback, not an upgrade: Kokoro is
# used whenever it has a voice for the language.
_ESPEAK = shutil.which("espeak-ng") or next(
    (p for p in (r"C:\Program Files\eSpeak NG\espeak-ng.exe",
                 r"C:\Program Files (x86)\eSpeak NG\espeak-ng.exe",
                 "/usr/bin/espeak-ng", "/opt/homebrew/bin/espeak-ng")
     if os.path.exists(p)), None)

# Whisper's two-letter codes, mapped to espeak-ng's voice names. They mostly
# agree; these are the ones that do not, plus the non-Latin cases where
# espeak-ng wants a base voice rather than a region (zh->cmn is Mandarin).
_ESPEAK_VOICE = {
    "zh": "cmn", "yue": "yue", "nb": "nn", "nn": "nn",
    "he": "he", "iw": "he", "jv": "jv",
}
_espeak_rate = 22050


def espeak_voice_for(lang: str | None) -> str | None:
    """The espeak-ng voice for a language, or None if it cannot speak it."""
    if not lang or not _ESPEAK:
        return None
    code = lang.lower().replace("_", "-").split("-")[0]
    if len(code) == 3:            # whisper sometimes reports 'tam', 'hin'
        return None               # let espeak guess from the full code instead
    return _ESPEAK_VOICE.get(code, code)


def _stream_espeak(text: str, lang: str | None):
    """One sentence -> int16 PCM at espeak's rate, via a UTF-8 temp file.

    Passing non-Latin text as a command-line argument or on stdin produces
    SILENCE on Windows (measured: Tamil, Chinese, Japanese, Korean, Arabic,
    Russian and Thai all came back with peak amplitude 0 and 0.35s of empty
    audio; only German, being Latin-1, worked through argv). Writing the
    sentence to a UTF-8 file and using -f is the one form that works for
    every script -- hence the temp file rather than the obvious subprocess
    call.
    """
    import wave as _wave
    voice = espeak_voice_for(lang)
    if not (voice and text.strip()):
        return
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="bt-espeak-")
    try:
        # os.fdopen is inside the try on purpose: if it itself raises, the
        # raw descriptor from mkstemp would never be closed.
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        try:
            rate = float(CFG.get("speed") or 1.0)
        except (TypeError, ValueError):
            rate = 1.0
        wpm = int(160 / max(rate, 0.5))
        proc = subprocess.run(
            [_ESPEAK, "-v", voice, "-s", str(wpm), "-f", path, "--stdout"],
            capture_output=True, timeout=60)
        if not proc.stdout:
            return
        with _wave.open(io.BytesIO(proc.stdout), "rb") as w:
            if w.getnchannels() > 1:      # espeak can emit stereo; we play mono
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                pcm = pcm.reshape(-1, w.getnchannels()).mean(axis=1).astype(np.int16)
            else:
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            sr = w.getframerate()
        if pcm.size:
            global _espeak_rate
            _espeak_rate = sr
            yield pcm
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def warm(lang: str | None = None) -> str:
    """Load (and cache) the Kokoro pipeline for a language's voice.

    Returns the voice name to speak with. The voice name's first letter IS
    the language pipeline: a=American English, b=British English,
    e/f/h/i/j/p/z = the other shipped languages. bm_lewis -> 'b'.

    Pipelines are cached per language letter, so switching languages does
    not reload anything already loaded; only a genuinely new language pays
    the startup cost, once.
    """
    global _pipe
    voice = voice_for(lang)
    code = (voice or "bm_lewis")[0]
    with _pipe_lock:
        if _pipe is None:
            _ensure_espeak()
            # Before kokoro makes this run's scratch dirs, clear the ones
            # earlier runs could not clean up on their way out.
            _sweep_orphan_espeak_tempdirs()
        if code not in _pipes:
            from kokoro import KPipeline
            log(f"[mouth] loading kokoro (lang '{code}', voice {voice})...")
            _pipes[code] = KPipeline(lang_code=code)
            log("[mouth] voice ready")
        _pipe = _pipes[code]
    return voice


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_RE.split(text.strip()) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _stream_kokoro(text: str):
    """One sentence -> int16 PCM chunks at 24kHz, in-process."""
    voice = warm(detect_language(text, _turn_lang()))
    pipe = _pipe
    try:
        speed = float(CFG.get("speed") or 1.0)
    except (TypeError, ValueError):
        speed = 1.0
    for _, _, audio in pipe(text, voice=voice, speed=speed):
        a = np.asarray(audio, dtype=np.float32)
        if a.size:
            yield (np.clip(a, -1.0, 1.0) * 32767).astype(np.int16)


def _stream_elevenlabs(text: str, timeout: float):
    """ElevenLabs -> ffmpeg streaming decode -> int16 PCM at 44.1kHz.

    THE ELEVENLABS DOCTRINE, learned the expensive way:
    - fetch mp3_44100_128 and decode locally (raw 44.1k PCM needs their
      Pro tier; the mp3 decode hides inside network wait anyway)
    - turbo model, stability 0.5, similarity 0.75
    - never the multilingual model for English, never style above 0 —
      both make delivery slow and dull
    - their site previews are MASTERED demo clips; raw API output never
      matches them, so master locally (the ffmpeg chain in config)
    ffmpeg reads stdin as we feed it, so playback still starts before
    synthesis finishes."""
    import subprocess

    import httpx

    el = CFG["elevenlabs"]
    key = _get_elevenlabs_key()
    url = (f"https://api.elevenlabs.io/v1/text-to-speech/"
           f"{el['voice_id']}/stream?output_format=mp3_44100_128")
    proc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "quiet", "-i", "pipe:0",
         "-af", el["master"],
         "-f", "s16le", "-ar", str(EL_RATE), "-ac", "1", "pipe:1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    feed_error: list = []

    def _feed():
        try:
            with httpx.stream("POST", url, headers={"xi-api-key": key},
                              json={"text": text, "model_id": el["model"],
                                    "voice_settings": {
                                        "stability": 0.5,
                                        "similarity_boost": 0.75}},
                              timeout=timeout) as r:
                r.raise_for_status()
                for chunk in r.iter_bytes(chunk_size=4096):
                    proc.stdin.write(chunk)
        except Exception as e:
            feed_error.append(e)
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    t = threading.Thread(target=_feed, daemon=True)
    t.start()
    carry = b""
    got_audio = False
    while True:
        data = proc.stdout.read(8820)
        if not data:
            break
        data = carry + data
        usable = len(data) - (len(data) % 2)
        carry = data[usable:]
        if usable:
            got_audio = True
            yield np.frombuffer(data[:usable], dtype=np.int16)
    proc.wait(timeout=10)
    if feed_error and not got_audio:
        raise feed_error[0]


_el_key_cache: str | None = None


def _key_slot() -> str:
    """The credential-store entry name, so someone who already keeps a key
    under their own name points at it instead of storing a second copy."""
    return str(CFG["elevenlabs"].get("key_slot") or "backtalk-elevenlabs")


def _get_elevenlabs_key() -> str:
    """The API key, from the most secure store available — NEVER from a
    file in this repo. Lookup order:
      1. macOS Keychain, item `backtalk-elevenlabs` by default (change it
         with elevenlabs.key_slot) — seed it once with:
         security add-generic-password -a "$USER" -s backtalk-elevenlabs -T /usr/bin/security -w
         (it prompts for the secret; -T lets this code read it without a
         GUI prompt every launch)
      2. Linux secret-tool (libsecret):
         secret-tool store --label backtalk service backtalk-elevenlabs
      3. the ELEVENLABS_API_KEY environment variable — the last-resort
         fallback, and the only option on Windows for now. Know the
         tradeoff: an export line in a shell profile is a plaintext key
         on disk, which is exactly what the keychain path avoids."""
    global _el_key_cache
    if _el_key_cache is not None:
        return _el_key_cache
    import subprocess
    key = ""
    try:
        if sys.platform == "darwin":
            r = subprocess.run(["security", "find-generic-password",
                                "-s", _key_slot(), "-w"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                key = r.stdout.strip()
        elif sys.platform.startswith("linux"):
            from shutil import which
            if which("secret-tool"):
                r = subprocess.run(["secret-tool", "lookup", "service",
                                    _key_slot()],
                                   capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    key = r.stdout.strip()
    except Exception:
        pass
    _el_key_cache = key or os.environ.get("ELEVENLABS_API_KEY", "")
    return _el_key_cache


def _elevenlabs_ready() -> bool:
    el = CFG["elevenlabs"]
    return bool(el.get("enabled") and el.get("voice_id")
                and _get_elevenlabs_key())


def synth_stream(text: str, timeout: float = 30.0):
    """One sentence -> yields (sample_rate, pcm_chunk) as the TTS
    renders. ElevenLabs when configured, Kokoro otherwise - and Kokoro
    as the fallback on ANY ElevenLabs failure. Degrade, never mute.

    When the turn's language has no Kokoro voice at all, espeak-ng speaks
    it in its own language rather than being read in an English accent.
    """
    lang = _turn_lang()
    if _elevenlabs_ready():
        try:
            for pcm in _stream_elevenlabs(text, timeout):
                yield EL_RATE, pcm
            return
        except Exception as e:
            log(f"[mouth] elevenlabs failed ({str(e)[:60]}) - "
                f"falling back to {CFG['voice']}")

    spoken = detect_language(text, lang)
    if spoken and spoken not in kokoro_langs() and espeak_voice_for(spoken):
        try:
            chunks = list(_stream_espeak(text, spoken))
            if chunks:
                log(f"[mouth] {spoken} has no kokoro voice - espeak-ng "
                    f"({espeak_voice_for(spoken)}) instead of an english accent")
                yield _espeak_rate, chunks[0]
                return
        except Exception as e:
            log(f"[mouth] espeak-ng failed ({str(e)[:60]}) - using kokoro")

    for pcm in _stream_kokoro(text):
        yield KOKORO_RATE, pcm


class Mouth:
    def __init__(self):
        from backtalk.ducking import Ducker
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._speaking = threading.Event()
        # The one persistent output stream (audio law #1).
        # Worker-thread-only — never touch from other threads.
        self._out: sd.OutputStream | None = None
        self._out_rate: int | None = None
        self.ducker = Ducker()  # public: PTT ducks for the USER's voice too
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    @property
    def speaking(self) -> bool:
        return self._speaking.is_set()

    def say(self, text: str):
        """Queue text (split to sentences) for speech."""
        for s in split_sentences(text):
            self._q.put((s, None))

    def say_chunk(self, text: str, directions=None):
        """Queue text as ONE TTS request, no sentence splitting — fuller
        chunks get livelier prosody (single short sentences come out
        dull).

        `directions` are the stage directions this chunk carried. They are
        published on the signal bus when this chunk's audio STARTS, which
        is why they travel with it instead of firing at parse time."""
        text = text.strip()
        if text:
            self._q.put((text, directions or None))

    def shut_up(self):
        """Barge-in: stop current playback and flush everything queued."""
        self._stop.set()
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def shutdown(self):
        """Exit path: stop playback and restore the music SYNCHRONOUSLY
        (the debounced restore timer dies with the process otherwise)."""
        self.shut_up()
        self.ducker.restore_now()

    def wait_done(self, timeout: float | None = None):
        """Block until the queue is drained and playback finished."""
        import time
        deadline = None if timeout is None else time.time() + timeout
        while (not self._q.empty()) or self._speaking.is_set():
            time.sleep(0.05)
            if deadline and time.time() > deadline:
                return

    def _run(self):
        from backtalk import signals
        while True:
            item = self._q.get()
            sentence, directions = item if isinstance(item, tuple) else (item, None)
            if not sentence:
                continue
            self._stop.clear()
            self._speaking.set()
            self.ducker.speech_start()
            signals.static_stop()     # thinking sound dies when speech starts
            signals.set_state("speaking")
            try:
                self._play_stream(sentence, directions)
            except Exception as e:
                log(f"[mouth] synth/play error: {e}")
            finally:
                if self._q.empty():
                    self._speaking.clear()
                    # The reply has genuinely stopped talking, as opposed to
                    # the gap between two sentences of the same reply.
                    # The caption goes only here, not when the sentences were
                    # queued: the queue drains well after the model is done.
                    signals.caption_clear()
                    signals.reply_done()
                    self.ducker.speech_end()
                    signals.set_state("idle")

    def _get_out(self, rate: int) -> sd.OutputStream:
        """The long-lived stream (audio law #1). Reopened only when the
        sample rate changes (ElevenLabs 44.1k <-> Kokoro 24k fallback:
        rare, costs at most one blip on the switch)."""
        if self._out is not None and self._out_rate == rate:
            # Guarded, because the stream can die UNDER us: the ears
            # rebuild the whole audio system to recover from a device
            # change (see ears._reopen_after_device_change), and that
            # closes every open stream including this one. Touching a
            # dead stream raises rather than returning False, so the
            # check has to be the try, not an `if`. Falling through
            # rebuilds it, which is what the rest of this method does.
            try:
                if not self._out.active:
                    self._out.start()
                return self._out
            except Exception:
                log("[mouth] the output stream went away, reopening")
        self._drop_out()
        self._out = sd.OutputStream(samplerate=rate, channels=1, dtype="int16")
        self._out_rate = rate
        self._out.start()
        return self._out

    def _cut(self):
        """Barge-in cut: stop feeding audio and pad the line with a beat
        of silence — the stream itself NEVER stops (an abort+restart here
        re-triggers the onset blip on latch-happy audio setups). Cost:
        the device buffer (~0.1s) plays out after the kill order — half a
        syllable of tail."""
        try:
            zeros = np.zeros(2205, dtype=np.int16)
            for _ in range(3):
                self._out.write(zeros)
        except Exception:
            self._drop_out()

    def _drop_out(self):
        """Close and forget the stream — the next sentence reopens
        fresh. The self-heal path for device errors (interface
        unplugged, audio mixer restarted)."""
        if self._out is not None:
            try:
                self._out.close(ignore_errors=True)
            except Exception:
                pass
        self._out = None
        self._out_rate = None

    def _play_stream(self, sentence: str, directions=None, block: int = 2205,
                     prebuffer_s: float = 0.75):
        """Stream-synthesize and play with the head-start buffer (audio
        law #2). stop() reacts ~50ms. The sample rate comes from
        whichever engine actually answered."""
        from backtalk import signals
        gen = synth_stream(sentence)
        head: list = []
        banked = 0
        rate = None
        for rate_, pcm in gen:
            rate = rate_
            head.append(pcm)
            banked += len(pcm)
            if banked >= int(rate * prebuffer_s):
                break
        if rate is None:
            return
        try:
            out = self._get_out(rate)
            # AUDIO STARTS HERE: the head buffer is full and the first write
            # is next. Publishing now is what puts a screen cue on the spoken
            # word rather than seconds ahead of it.
            if directions:
                from backtalk import signals as _sig
                _sig.direction(directions)

            def _write(pcm):
                for i in range(0, len(pcm), block):
                    if self._stop.is_set():
                        return False
                    out.write(pcm[i:i + block])
                    # Re-check after the blocking write: a barge-in
                    # landing mid-block must not let feed_waveform
                    # re-assert "speaking" over a fresh "listening".
                    if self._stop.is_set():
                        return False
                    signals.feed_waveform(pcm[i:i + block])
                return True
            for pcm in head:
                if not _write(pcm):
                    self._cut()
                    return
            for _, pcm in gen:
                if not _write(pcm):
                    self._cut()
                    return
        except Exception:
            self._drop_out()
            raise


if __name__ == "__main__":
    m = Mouth()
    m.say(sys.argv[1] if len(sys.argv) > 1 else
          "Voice check. The mouth is alive, and it is very good to be heard.")
    m.wait_done(timeout=60)
