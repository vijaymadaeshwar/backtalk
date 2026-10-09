# backtalk tests

There is no pytest here on purpose: every test is a standalone script
that prints `ok`/`FAIL` lines and exits nonzero on failure. `run_all.py`
finds them, runs each in its own process, and summarises.

    uv run python tests/run_all.py          the standard suites
    uv run python tests/run_all.py --fast   skip the model-loading suites (CI)
    uv run python tests/run_all.py --all    include the live/e2e suites

Some suites are pure control-flow with a faked mic and transcriber and
run in seconds. Others run the real Whisper decoder, the real brain over
the opencode server, and the real TTS engines, and take minutes -- those
are excluded from `--fast`. `--all` adds the suites that need the live
stack running (brain on 4599, face on 8790).

| script | what it proves |
| --- | --- |
| `test_wake.py` | Wake phrases are recognised however whisper spells or punctuates them, stripped without damaging the command, and the shipped phrases are Latin-only. |
| `test_endpointing.py` | The open-mic endpointer opens and closes on schedule, and the loudness gate drops quiet noise before whisper sees it. |
| `test_ptt.py` | Hold-to-talk timing: the release callback fires before transcription, taps are ignored, the time cap still proceeds. |
| `test_encoding.py` | Every text file is clean UTF-8: no BOM, no double-encoding. |
| `test_journal.py` | What the session journal writes, what it refuses to write, and what it does when the brain is slow or broken. |
| `test_bus_park.py` | The bus is never left saying `speaking` by a voice that is gone. |
| `test_turn_errors.py` | A failed turn always parks the bus and tells the listener. |
| `test_stt_langs.py` | A real English or Tamil sentence survives the trip through Whisper (heavy). |
| `test_espeak_fallback.py` | Every reply reaches the one English voice, and espeak-ng carries it when Kokoro cannot (heavy). |
| `test_e2e.py` | One whole turn: speech -> STT -> brain -> TTS -> caption and language on the bus (live). |
| `test_live_path.py` | A manual driver for the live-data path; pass it a question (live). |

## What these found

Worth keeping, because each one looked fine from the outside and was not:

- An English `stt_prompt` made Whisper transcribe a Japanese sentence as
  "Hello, Memo, please open your intestines." A multilingual prompt fixed
  it -- and a prompt carrying a wake phrase plus a command later made
  Whisper invent that exact turn out of room noise and Seyon executed it.
  The prompt is empty now.
- On Windows, espeak-ng produces pure **silence** for non-Latin scripts
  when the text is passed as argv or on stdin. It only works via `-f` on a
  UTF-8 file, which is why `mouth._stream_espeak` writes a temp file.
- The old language detector returned English for Tamil, so a Tamil voice
  never fired. The detector, the per-language voices, and MMS are removed:
  the ear hears anything, the mouth answers in English, and these suites
  pin that so a language branch cannot come back.
- `brain.WarmBrain.ask_stream` yields nothing at all unless `start()` has
  run first. An E2E test that skips it reports an empty reply and looks
  like a brain failure rather than a harness mistake.
