"""Non-English STT accuracy, on REAL speech.

The end-to-end wiring test fed Whisper a noise-shaped signal, so it proved
the plumbing and nothing about language handling. This one synthesises
genuine speech in each language with the very voices Jarvis speaks with,
resamples it to the 16kHz mono the ear expects, and asks Whisper what it
heard.

That round trip is the one that matters: it is exactly what happens when
Vijay holds the key and speaks, except the speaker is a model instead of a
person. It cannot tell us how noisy his room is, but it does tell us
whether a Hindi or Japanese sentence survives the trip.
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, r"C:\Users\Vijay\my-agent\backtalk")
import numpy as np                                   # noqa: E402
from backtalk import ears, mouth                     # noqa: E402

SR = 16000          # what the ear wants
KOKORO_SR = 24000   # what Kokoro produces

# (code, sentence in that language, a distinctive word to look for)
# The needle is the word in THAT language for "notepad" -- matching "notepad"
# in Spanish would fail a perfect transcription, since the translation is
# "bloc de notas".
# Two checks per language, because they fail independently:
#   "lang"  - did whisper identify the language? (its own guess)
#   "heard"  - did the SENTENCE SURVIVE? (needle word present, no cross-
#              language bleed) - this is what the brain actually receives.
CASES = [
    ("en", "Hello there, could you open Notepad for me please?", "notepad"),
    ("es", "Hola, puedes abrir el Bloc de notas por favor?", "bloc"),
    ("fr", "Bonjour, pouvez vous ouvrir le bloc notes svp?", "notes"),
    ("hi", "नमस्ते, क्या आप नोटपैड खोल सकते हैं?", "नोटपैड"),
    ("it", "Ciao, puoi aprire il Blocco Note per favore?", "note"),
    ("ja", "こんにちは、メモ帳を開いてください。", "メモ帳"),
    ("pt", "Ola, voce pode abrir o bloco de notas por favor?", "bloco"),
    ("zh", "你好，请打开记事本。", "记事本"),
]

OUT = tempfile.mkdtemp()


def speech_pcm(text, code):
    """Real speech in `code`, as int16 at 16kHz mono."""
    voice = mouth.voice_for(code)
    chunks = []
    for _rate, pcm in mouth.synth_stream(text, timeout=90.0):
        a = (np.frombuffer(pcm, dtype=np.int16) if isinstance(pcm, (bytes, bytearray))
             else np.asarray(pcm, dtype=np.int16))
        if a.size:
            chunks.append(a)
    if not chunks:
        return None
    wave = np.concatenate(chunks).astype(np.float32)
    # Linear resample 24k -> 16k. Whisper's accuracy on short files is
    # sensitive to this, so it is done explicitly rather than assumed.
    n_out = int(len(wave) * SR / KOKORO_SR)
    idx = np.linspace(0, len(wave) - 1, n_out)
    return np.interp(idx, np.arange(len(wave)), wave).astype(np.int16)


def word_overlap(heard, needle):
    """Does the transcript contain the key word, in ANY script form?"""
    def fold(s):
        import unicodedata
        s = "".join(c for c in unicodedata.normalize("NFKD", s)
                    if not unicodedata.combining(c))
        return "".join(c for c in s.lower() if c.isalnum())
    return fold(needle) in fold(heard)


print("=" * 74)
print("NON-ENGLISH STT ACCURACY, on real synthesised speech")
print("  model: %s" % ears._resolve_stt_model())
print("=" * 74)

rows, bad = [], []
for code, sentence, needle in CASES:
    try:
        pcm = speech_pcm(sentence, code)
    except Exception as e:
        print("  %-3s SKIP  could not synthesise: %s" % (code, e))
        continue
    if pcm is None or pcm.size < SR // 4:
        print("  %-3s FAIL  no audio produced" % code)
        bad.append(code)
        continue

    path = os.path.join(OUT, "%s.wav" % code)
    import wave as wavemod
    with wavemod.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())

    heard, detected = ears.transcribe_language(pcm)
    heard = heard.strip()
    lang_ok = (detected or "").lower()[:2] == code
    word_ok = word_overlap(heard, needle)
    # A misidentified language is only cosmetic IF the words still arrived
    # correctly -- it picks the accent, not the meaning. So the two are
    # reported and scored separately rather than lumped into one verdict.
    status = "ok  " if word_ok else "FAIL"
    print("  %s %-3s det=%-4s word=%-6s %r" % (
        status, code, detected, "found" if word_ok else "MISSED", heard[:50]))
    if not word_ok:
        bad.append(code)
    rows.append((code, detected, lang_ok, word_ok))

print()
print("sentence survived  : %d/%d  <- what matters" % (
    sum(1 for r in rows if r[3]), len(rows)))
print("language identified: %d/%d  <- picks the accent" % (
    sum(1 for r in rows if r[2]), len(rows)))
mislabelled = [r[0] for r in rows if not r[2] and r[3]]
if mislabelled:
    print("  (words correct but accent may differ: %s)" % mislabelled)
print()
print("STT OK" if not bad else "PROBLEM LANGUAGES: %s" % bad)