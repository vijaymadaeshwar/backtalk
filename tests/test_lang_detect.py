"""detect_language decides which language a reply is written in, which
decides which VOICE speaks it. Pure text sniffing -- no model, no audio,
so it runs in milliseconds.

Worth having because this is where the Portuguese bug lived: "por favor"
was a Spanish marker checked before the stopword score, so a Portuguese
sentence ending in it was declared Spanish, got a Spanish voice, and the
test's "Portuguese" audio was Spanish. That kind of bug only surfaced
through the slow audio suite. This pins it directly."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtalk.mouth import detect_language, espeak_voice_for, _ESPEAK  # noqa: E402


def check(name, cond, detail=""):
    print("    %s %s%s" % ("ok  " if cond else "FAIL", name,
                           ("  " + str(detail)) if detail else ""))
    if not cond:
        FAILURES.append(name)
    return cond


FAILURES = []

# (text, expected). Each is unmistakably the language it claims to be.
CASES = [
    # The regression: Portuguese ending in the shared phrase "por favor".
    ("Ola, voce pode abrir o bloco de notas por favor?", "pt"),
    ("Bom dia, tudo bem com voce?", "pt"),
    # Spanish must still win, including when IT uses "por favor".
    ("Hola, puedes abrir el bloc de notas por favor?", "es"),
    ("Hola, muchas gracias por favor.", "es"),
    ("Bonjour, pouvez vous ouvrir le bloc notes svp?", "fr"),
    ("Guten Tag, koennen Sie die Datei bitte oeffnen?", "de"),
    ("Ciao, puoi aprire il Blocco Note per favore?", "it"),
    ("Hello there, could you open Notepad for me please?", "en"),
    # Non-Latin scripts, settled by script before any word list.
    ("こんにちは、メモ帳を開いてください。", "ja"),
    ("你好，请打开记事本。", "zh"),
    ("नमस्ते, क्या आप नोटपैड खोल सकते हैं?", "hi"),
    ("நமஸ்தே, எப்படி இருக்கிறீர்கள்?", "ta"),
    # Nothing to sniff: English is the safe fallback, never a guess.
    ("", "en"),
]


def test_cases():
    print("\n--- each sentence is read as the language it is ---")
    for text, want in CASES:
        got = detect_language(text)
        check("%-3s -> %s" % (want, repr(text[:34])),
              got == want, detail="got %r" % got)


def test_hint_wins():
    print("\n--- the ear's hint outranks the text ---")
    check("hint fr beats English text",
          detect_language("Hello there", hint="fr") == "fr",
          detail=detect_language("Hello there", hint="fr"))
    check("hint es beats English text",
          detect_language("open the file", hint="es") == "es")
    # Whisper's codes are usually two letters but not always: 'yue' for
    # Cantonese comes whole, and it used to be truncated to 'yu' -- no
    # language, so Cantonese fell through to an English accent.
    check("whisper's 'yue' stands for Cantonese",
          detect_language("hello there", hint="yue") == "yue",
          detail=detect_language("hello there", hint="yue"))
    if _ESPEAK:
        check("espeak really can speak it",
              espeak_voice_for("yue") == "yue", detail=espeak_voice_for("yue"))
    else:
        print("    skip  espeak_voice_for (espeak-ng not installed)")
    # ISO 639-3 forms shorten to the right two-letter code, not nothing.
    check("'hin' lands on Hindi",
          detect_language("hello there", hint="hin") == "hi",
          detail=detect_language("hello there", hint="hin"))
    check("'tam' lands on Tamil",
          detect_language("hello there", hint="tam") == "ta",
          detail=detect_language("hello there", hint="tam"))


def test_never_raises():
    print("\n--- odd input is handled, not raised ---")
    for text in (None, "   ", "\n\t", "🙂", "a"):
        try:
            detect_language(text)
            ok = True
            detail = ""
        except Exception as e:            # noqa: BLE001
            ok, detail = False, "%s: %s" % (type(e).__name__, e)
        check("detect_language(%r)" % (text,), ok, detail=detail)


print("=" * 66)
print("LANGUAGE DETECTION")
print("=" * 66)

test_cases()
test_hint_wins()
test_never_raises()

print("\n" + "=" * 66)
print("LANG OK" if not FAILURES else "LANG FAILURES: %s" % FAILURES)
sys.exit(1 if FAILURES else 0)
