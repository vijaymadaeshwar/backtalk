"""Run the backtalk test suites in one command.

Each test is a standalone script that exits nonzero on failure -- there
is no pytest here on purpose. This finds them, runs each in its own
process, and summarises.

    uv run python tests/run_all.py          the standard suites
    uv run python tests/run_all.py --all    also the live / e2e suites
    uv run python tests/run_all.py --fast   skip anything that loads a model

--fast is what CI runs: no whisper, no kokoro, no audio hardware.
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Suites that download a model, start a TTS engine, or drive the whole
# voice line: too slow or too environment-bound for CI.
HEAVY = {"test_stt_langs.py", "test_espeak_fallback.py"}
# Suites that need the live stack running. Opt in with --all.
LIVE = {"test_e2e.py", "test_live_path.py"}


def discover(fast: bool, every: bool) -> list[str]:
    chosen = []
    for path in sorted(HERE.glob("test_*.py")):
        name = path.name
        if name in LIVE and not every:
            continue
        if name in HEAVY and fast:
            continue
        chosen.append(name)
    return chosen


def main(argv: list[str]) -> int:
    fast = "--fast" in argv
    every = "--all" in argv
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    chosen = discover(fast, every)
    if not chosen:
        print("no suites selected")
        return 1
    tag = " [fast]" if fast else (" [all]" if every else "")
    print(f"running {len(chosen)} suite(s){tag}\n")
    failed = []
    for name in chosen:
        print("=" * 66)
        print(name)
        print("=" * 66)
        proc = subprocess.run([sys.executable, str(HERE / name)], env=env)
        if proc.returncode != 0:
            failed.append(name)
        print()
    print("=" * 66)
    passed = len(chosen) - len(failed)
    print(f"{passed}/{len(chosen)} suites passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print("ALL SUITES PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
