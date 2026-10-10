"""Run the backtalk test suites in one command.

Each test is a standalone script that exits nonzero on failure -- there
is no pytest here on purpose. This finds them, runs each in its own
process, and summarises.

    uv run python tests/run_all.py          the standard suites
    uv run python tests/run_all.py --all    also the live / e2e suites
    uv run python tests/run_all.py --fast   skip anything that loads a model
    uv run python tests/run_all.py --coverage   measure backtalk coverage

--fast is what CI runs: no whisper, no kokoro, no audio hardware.
--coverage runs each suite under coverage.py and fails under the floor.
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
# The coverage floor the fast suites must clear. The number is what the
# offline suites actually reach for backtalk/ (measured 71% in late 2026);
# it is a ratchet, not a target. The uncovered remainder is what a fast,
# headless run cannot reach: the live event loop and the POSIX tty reader
# (main.amain, _typed_reader), the whisper model and microphone (ears),
# the real audio sinks (mouth's kokoro/elevenlabs streaming), the brain's
# provider glue, and the keychain/secret-tool lookups. Raise this whenever
# the real number rises -- never lower it to pass.
COVERAGE_FLOOR = 69


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
    cov = "--coverage" in argv
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    chosen = discover(fast, every)
    if not chosen:
        print("no suites selected")
        return 1
    runner = [sys.executable]
    if cov:
        subprocess.run([sys.executable, "-m", "coverage", "erase"], env=env)
        runner += ["-m", "coverage", "run", "--append"]
    tag = " [fast]" if fast else (" [all]" if every else "")
    print(f"running {len(chosen)} suite(s){tag}\n")
    failed = []
    for name in chosen:
        print("=" * 66)
        print(name)
        print("=" * 66)
        proc = subprocess.run(runner + [str(HERE / name)], env=env)
        if proc.returncode != 0:
            failed.append(name)
        print()
    print("=" * 66)
    passed = len(chosen) - len(failed)
    print(f"{passed}/{len(chosen)} suites passed")
    if cov:
        print("=" * 66)
        rep = subprocess.run(
            [sys.executable, "-m", "coverage", "report",
             f"--fail-under={COVERAGE_FLOOR}"], env=env)
        if rep.returncode != 0:
            failed.append(f"coverage<{COVERAGE_FLOOR}%")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print("ALL SUITES PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
