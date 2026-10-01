# backtalk tests

Real checks against real models. None of these are mocks: they run the
actual Whisper decoder, the actual brain over the real OpenCode server,
and the actual TTS engines, then read the face's state back over HTTP.

Run them with the stack up (brain on 4599, face on 8790):

    .venv\Scripts\python.exe tests\test_stt_langs.py
    .venv\Scripts\python.exe tests\test_espeak_fallback.py
    .venv\Scripts\python.exe tests\test_e2e.py

They take real time — minutes, not seconds — because speech recognition on
CPU and a 120B-parameter model are both slow on purpose.

| script | what it proves |
| --- | --- |
| `test_stt_langs.py` | A real sentence in each of 8 languages survives the trip through Whisper. Synthesises the speech with the same voices Jarvis uses, resamples to 16kHz, and checks the transcription kept its meaning. |
| `test_espeak_fallback.py` | Languages Kokoro has no voice for (Tamil, Korean, Arabic, Russian, Thai) are spoken by espeak-ng in their own language, and languages Kokoro *does* have stay on Kokoro. |
| `test_e2e.py` | One whole turn: speech → STT → brain → TTS → caption and language on the bus → what the face displays. |
| `test_live_path.py` | The live-data path, that is the reason the lookup exists at all. |

## What these found

Worth keeping, because each one looked fine from the outside and was not:

- An English `stt_prompt` made Whisper transcribe a Japanese sentence as
  "Hello, Memo, please open your intestines." A multilingual prompt fixed it.
- On Windows, espeak-ng produces pure **silence** for non-Latin scripts when
  the text is passed as an argv argument or on stdin. It only works via
  `-f` on a UTF-8 file. That is why `mouth._stream_espeak` writes a temp file.
- `mouth.detect_language` returned English for Tamil, so the espeak-ng
  fallback never fired and Tamil was read in a British accent for months.
- `brain.WarmBrain.ask_stream` yields nothing at all unless `start()` has
  run first. An E2E test that skips it reports an empty reply and looks
  like a brain failure rather than a harness mistake.