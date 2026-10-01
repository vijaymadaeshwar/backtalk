"""Does the espeak-ng fallback actually speak the languages Kokoro cannot?

The claim being tested: Tamil, Korean, Arabic, Russian and Thai used to be
read in a British English accent because Kokoro ships no voice for them,
and now are spoken in their own language. Each case is run twice - once
with the turn language set, once without - because the fallback keys off the
turn language and must not fire on an ordinary English turn.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np                                   # noqa: E402
from backtalk import mouth                          # noqa: E402

NO_KOKORO = [
    # German is the regression that motivated this list. It HAS an entry in
    # the voices table (an English fallback), so an earlier version asked
    # "is this language in voices?" and answered yes -- reading German aloud
    # in a British accent instead of speaking it. Membership must be judged
    # against what Kokoro actually ships, not against that table.
    ("de", "Guten Morgen, ich habe eine Frage fuer dich."),
    ("ta", "வணக்கம், தயவுசெய்து நோட்டபாட் திறக்க முடியுமா?"),
    ("ko", "안녕하세요, 메모장을 열어 주세요."),
    ("ar", "مرحبا، افتح المفكرة من فضلك."),
    ("ru", "Привет, открой блокнот пожалуйста."),
    ("th", "สวัสดี กรุณาเปิดโน้ตแพด"),
]

HAS_KOKORO = [
    ("es", "Hola, puedes abrir el Bloc de notas por favor?"),
    ("ja", "こんにちは、メモ帳を開いてください。"),
    ("hi", "नमस्ते, क्या आप नोटपैड खोल सकते हैं?"),
]

print("espeak-ng found at: %s" % mouth._ESPEAK)
print("voice mapping: %s" % {k: mouth.espeak_voice_for(k)
                             for k in ("ta", "ko", "ar", "ru", "th", "zh", "ja")})
print()
print("--- languages with NO kokoro voice (must use espeak-ng) ---")
bad = []
for code, text in NO_KOKORO:
    mouth.set_turn_language(code)
    chunks = [(r, np.frombuffer(p, dtype=np.int16))
              for r, p in mouth.synth_stream(text, timeout=60.0)]
    pcm = np.concatenate([c for _r, c in chunks]) if chunks else np.zeros(0, np.int16)
    rate = chunks[0][0] if chunks else 0
    secs = len(pcm) / rate if rate else 0
    peak = int(np.abs(pcm).max()) if pcm.size else 0
    ok = secs > 0.3 and peak > 500
    print("  %s %-3s rate=%-6d %.2fs peak=%-6d" % (
        "ok  " if ok else "FAIL", code, rate, secs, peak))
    if not ok:
        bad.append(code)

print("\n--- languages WITH a kokoro voice (must NOT use espeak-ng) ---")
for code, text in HAS_KOKORO:
    mouth.set_turn_language(code)
    chunks = [(r, np.frombuffer(p, dtype=np.int16))
              for r, p in mouth.synth_stream(text, timeout=90.0)]
    rate = chunks[0][0] if chunks else 0
    used = "kokoro" if rate == 24000 else "ESPEAK (wrong!)"
    ok = rate == 24000
    print("  %s %-3s rate=%-6d -> %s" % ("ok  " if ok else "FAIL", code, rate, used))
    if not ok:
        bad.append(code)

print("\n--- plain english turn (must stay on kokoro) ---")
mouth.set_turn_language("en")
chunks = [(r, np.frombuffer(p, dtype=np.int16))
          for r, p in mouth.synth_stream("Opening Notepad now.", timeout=90.0)]
rate = chunks[0][0] if chunks else 0
print("  %s rate=%d" % ("ok  " if rate == 24000 else "FAIL", rate))
if rate != 24000:
    bad.append("en")

print("\n--- turn language cleared (must fall back to kokoro) ---")
mouth.set_turn_language(None)
chunks = [(r, np.frombuffer(p, dtype=np.int16))
          for r, p in mouth.synth_stream("Opening Notepad now.", timeout=90.0)]
rate = chunks[0][0] if chunks else 0
print("  %s rate=%d" % ("ok  " if rate == 24000 else "FAIL", rate))
if rate != 24000:
    bad.append("no-turn-lang")

print("\nESPEAK FALLBACK OK" if not bad else "FAILURES: %s" % bad)