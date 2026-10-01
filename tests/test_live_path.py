"""Drive backtalk's real ask_stream so the live-data injection is exercised."""
import asyncio
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtalk.brain import WarmBrain  # noqa: E402


async def ask(q):
    brain = WarmBrain()
    await brain.start()
    t0 = time.time()
    out = []
    async for sentence in brain.ask_stream(q):
        out.append(sentence)
        if time.time() - t0 > 180:
            break
    return time.time() - t0, " ".join(out)


for question in sys.argv[1:]:
    took, reply = asyncio.run(ask(question))
    print(f"\n=== {took:.1f}s  {question}")
    print(reply[:700])
