"""main.py offline: the pure helpers, the reply batching, and the spoken
permission gate -- the parts of the live event loop that can be pinned
without a mic, a model, or an opencode server.

`amain()` itself is a live driver and stays uncovered here; everything it
calls that has a decision in it is exercised directly.
"""
from pathlib import Path
import asyncio
import os
import queue
import sys
import tempfile
import threading
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtalk import main as btmain  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'ok' if cond else 'FAIL'} {name}"
          + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILURES.append(name)


class FakeMouth:
    def __init__(self):
        self.chunks = []
        self.pending = []
        self.said = []

    def say_chunk(self, s, pending=None):
        self.chunks.append(s)
        self.pending.append(list(pending or []))

    def say(self, s):
        self.said.append(s)


class FakeBrain:
    def __init__(self, items):
        self.items = items
        self.got = None
        self.interrupted = False

    async def ask_stream(self, text):
        self.got = text
        for it in self.items:
            if isinstance(it, BaseException):
                raise it
            yield it

    async def interrupt(self):
        self.interrupted = True


def console():
    print("\n--- console_match: exact phrases only ---")
    check("a verb phrase matches",
          btmain.console_match("clear the session") == "clear")
    check("case and punctuation are ignored",
          btmain.console_match("Compact Context!") == "compact")
    check("hyphens fold to spaces",
          btmain.console_match("hands-free mode") == "micopen")
    check("an effort level matches",
          btmain.console_match("set effort to high") == "effort:high")
    check("a bare effort word matches",
          btmain.console_match("effort low") == "effort:low")
    check("ordinary speech never triggers a verb",
          btmain.console_match("please clear the table") is None)
    check("unknown phrase -> None", btmain.console_match("hello") is None)


def formatting():
    print("\n--- _fmt_tokens / _spoken_usage ---")
    check("small number", btmain._fmt_tokens(999) == "999 tokens")
    check("thousands", btmain._fmt_tokens(1000) == "about 1 thousand tokens",
          btmain._fmt_tokens(1000))
    check("millions", btmain._fmt_tokens(1_500_000)
          == "about 1.5 million tokens", btmain._fmt_tokens(1_500_000))

    sess = {"turns": 2, "out_tokens": 999, "cost": 2.0}
    usage = {"categories": [
        {"name": "System", "tokens": 1000},
        {"name": "Free space", "tokens": 5000},
        {"name": "Autocompact buffer", "tokens": 3000},
        {"name": "Messages", "tokens": 2000},
    ]}
    out = btmain._spoken_usage(sess, usage)
    check("turn count spoken", "2 turns this session" in out, out)
    check("tokens spoken", "999 tokens spoken out" in out, out)
    check("dollars spoken", "roughly 2 dollars" in out, out)
    check("free space excluded from the occupied total",
          "about 3 thousand tokens sitting in the context window" in out, out)

    one = btmain._spoken_usage({"turns": 1, "out_tokens": 500, "cost": 0.0},
                               {"categories": []})
    check("singular turn", "1 turn this session" in one, one)
    check("no cost -> no money clause", "cents" not in one and "dollars" not in one,
          one)

    weird = btmain._spoken_usage({"turns": 1, "out_tokens": 10, "cost": 0.0},
                                 mock.Mock())
    check("a broken usage object does not raise",
          "1 turn this session" in weird, weird)


def typed_input():
    print("\n--- _clean_typed / _join_paste / _typed_reader_pipe ---")
    check("gutter glyph scrubbed",
          btmain._clean_typed("▎ Hello") == "Hello")
    check("nested gutters scrubbed",
          btmain._clean_typed("│> quoted") == "quoted")
    check("plain text untouched", btmain._clean_typed("  hi  ") == "hi")
    check("paste joined to one line",
          btmain._join_paste("▎ one\n▎ two") == "one two")

    r, w = os.pipe()
    q: queue.Queue = queue.Queue()
    os.write(w, b"hello world\n\x1b[200~pasted\nlines\x1b[201~last line\n")
    os.close(w)
    t = threading.Thread(target=btmain._typed_reader_pipe, args=(q, r),
                         daemon=True)
    t.start()
    t.join(timeout=2)
    os.close(r)
    got = []
    while not q.empty():
        got.append(q.get_nowait())
    check("plain line read", "hello world" in got, got)
    check("paste collapsed into one message", "pasted lines" in got, got)
    check("line after the paste still read", "last line" in got, got)

    # a paste marker with no closing marker yet: hold it, then EOF.
    r2, w2 = os.pipe()
    q2: queue.Queue = queue.Queue()
    os.write(w2, b"\x1b[200~unfinished")
    os.close(w2)
    btmain._typed_reader_pipe(q2, r2)
    os.close(r2)
    check("an unterminated paste is held, not emitted", q2.empty())

    # a closed descriptor: the reader must return, not raise.
    r3, w3 = os.pipe()
    os.close(r3)
    os.close(w3)
    btmain._typed_reader_pipe(queue.Queue(), r3)
    check("a dead descriptor ends the reader quietly", True)

    import builtins
    q3: queue.Queue = queue.Queue()
    with mock.patch.object(builtins, "input",
                           side_effect=["  hello  ", "│> quoted", EOFError]):
        btmain._typed_reader_simple(q3)
    simple = []
    while not q3.empty():
        simple.append(q3.get_nowait())
    check("the simple reader scrubs and queues",
          simple == ["hello", "quoted"], simple)

    r4, w4 = os.pipe()
    os.write(w4, b"piped line\n")
    os.close(w4)
    q4: queue.Queue = queue.Queue()
    fake_stdin = mock.Mock()
    fake_stdin.fileno.return_value = r4
    with mock.patch.object(btmain.sys, "stdin", fake_stdin):
        btmain._typed_reader(q4)
    os.close(r4)
    piped = []
    while not q4.empty():
        piped.append(q4.get_nowait())
    check("a non-tty stdin is read as lines", piped == ["piped line"], piped)


def human_forms():
    print("\n--- _human_what / _full_detail ---")
    with mock.patch.object(btmain, "CFG", {"extra_dirs": ["C:/vault"]}):
        check("a vault markdown note is named",
              btmain._human_what("Write", {"file_path": "C:/vault/note.md"},
                                 None)
              == "create or change a note in your vault called note",
              btmain._human_what("Write",
                                 {"file_path": "C:/vault/note.md"}, None))
        check("an edit of a vault note says edit",
              btmain._human_what("Edit", {"file_path": "C:/vault/x.md"}, None)
              == "edit a note in your vault called x")
    check("an ordinary file is named, not pathed",
          btmain._human_what("Write", {"file_path": "C:/x/report.txt"}, None)
          == "create or change a file called report.txt")
    check("a bash command names the program",
          btmain._human_what("Bash", {"command": "git status"}, None)
          == "run a git command in the terminal")
    check("a chained command says so",
          btmain._human_what("Bash", {"command": "a && b"}, None)
          .endswith("with several chained parts"))
    check("a fetch names the host",
          btmain._human_what("WebFetch", {"url": "https://example.com/x"},
                             None) == "read a web page at example.com")
    ctx = mock.Mock(display_name="Todo", description="a task list")
    check("an unknown tool uses its display name",
          btmain._human_what("TodoWrite", {}, ctx) == "use the Todo tool")

    long_cmd = "x" * 130
    detail = btmain._full_detail("Bash", {"command": long_cmd}, None)
    check("a long command warns about hidden characters",
          "more characters" in detail and "Check the log" in detail, detail)
    check("a write shows the last two path segments",
          btmain._full_detail("Write", {"file_path": "a/b/report.txt"}, None)
          == "write the file b/report.txt",
          btmain._full_detail("Write", {"file_path": "a/b/report.txt"}, None))
    check("a fetch shows the url",
          btmain._full_detail("WebFetch", {"url": "https://x.example/p"},
                              None) == "fetch a web page: https://x.example/p")
    check("an unknown tool appends its description",
          btmain._full_detail("TodoWrite", {}, ctx)
          == "use Todo, a task list",
          btmain._full_detail("TodoWrite", {}, ctx))


def config_write():
    print("\n--- _write_config_key: session-only on a bad file ---")
    import backtalk.config as cfgmod
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "backtalk.json"
        path.write_text('{"keep": 1}')
        with mock.patch.object(btmain, "CFG", {"keep": 1}), \
             mock.patch.object(cfgmod, "CONFIG_PATH", path), \
             mock.patch.object(btmain, "log"):
            ok = btmain._write_config_key("new", "value")
            written = path.read_text()
        check("write reports success", ok)
        check("the other key survived",
              "keep" in written and "new" in written, written)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "missing.json"
        with mock.patch.object(btmain, "CFG", {}), \
             mock.patch.object(cfgmod, "CONFIG_PATH", path), \
             mock.patch.object(btmain, "log"):
            ok = btmain._write_config_key("a", 1)
        check("a missing file is created", ok and path.exists())

    bad = mock.Mock()
    bad.read_text.return_value = "not json at all"
    cfg = {"a": 0}
    with mock.patch.object(btmain, "CFG", cfg), \
         mock.patch.object(cfgmod, "CONFIG_PATH", bad), \
         mock.patch.object(btmain, "log") as log:
        ok = btmain._write_config_key("a", 1)
    check("an unparsable file is left alone (session-only)",
          ok is False and not bad.write_text.called)
    check("but the in-memory config still updated", cfg["a"] == 1, cfg)
    check("the failure was logged", log.called)

    ro = mock.Mock()
    ro.read_text.return_value = '{"a": 1}'
    ro.write_text.side_effect = OSError("read-only")
    with mock.patch.object(btmain, "CFG", {}), \
         mock.patch.object(cfgmod, "CONFIG_PATH", ro), \
         mock.patch.object(btmain, "log"):
        ok = btmain._write_config_key("a", 2)
    check("a read-only file fails softly", ok is False)


def speak_first_alone():
    print("\n--- speak_reply: first sentence alone, directions, on_reply ---")
    brain = FakeBrain(["Hello <<wave>> there.", "Second.", "Third.",
                       "Fourth."])
    mouth = FakeMouth()
    spoken: list = []
    with mock.patch.object(btmain, "signals") as sig, \
         mock.patch.object(btmain, "log"):
        asyncio.run(btmain.speak_reply(brain, mouth, "hi",
                                       on_reply=spoken.append))
    check("brain got the raw question", brain.got == "hi")
    check("first sentence ships alone",
          mouth.chunks[:1] == ["Hello there."], mouth.chunks)
    check("the direction rode with the first chunk",
          mouth.pending[0] == ["wave"], mouth.pending)
    check("the rest go in a two-sentence breath",
          mouth.chunks[1] == "Second. Third.", mouth.chunks)
    check("the trailing sentence is flushed",
          mouth.chunks[-1] == "Fourth.", mouth.chunks)
    check("on_reply saw each spoken chunk", spoken == mouth.chunks, spoken)
    check("a full reply leaves the bus alone",
          not sig.set_state.called and not sig.static_stop.called)


def speak_empty_and_errors():
    print("\n--- speak_reply: empty and failing turns park the bus ---")
    mouth = FakeMouth()
    with mock.patch.object(btmain, "signals") as sig, \
         mock.patch.object(btmain, "log"):
        asyncio.run(btmain.speak_reply(FakeBrain([]), mouth, "hi"))
    check("nothing spoken on an empty reply", not mouth.chunks and not mouth.said)
    check("an empty turn parks the bus",
          sig.static_stop.called and sig.caption_clear.called)
    states = [c.args[0] for c in sig.set_state.call_args_list]
    check("bus parked to idle", states == ["idle"], states)

    mouth2 = FakeMouth()
    with mock.patch.object(btmain, "signals") as sig2, \
         mock.patch.object(btmain, "log"):
        asyncio.run(btmain.speak_reply(FakeBrain([RuntimeError("boom")]),
                                       mouth, "hi"))
    check("a hard failure says so out loud",
          mouth.said and "could not reach" in mouth.said[0], mouth.said)
    check("and parks the bus", sig.static_stop.called)

    brain = FakeBrain(["One.", "Two.", RuntimeError("mid-stream")])
    mouth2 = FakeMouth()
    with mock.patch.object(btmain, "signals") as sig, \
         mock.patch.object(btmain, "log"):
        asyncio.run(btmain.speak_reply(brain, mouth2, "hi"))
    check("a sentence already in hand is still spoken",
          mouth2.chunks == ["One.", "Two."], mouth2.chunks)
    check("no error line when something was already said",
          mouth2.said == [], mouth2.said)

    brain = FakeBrain(["One.", asyncio.CancelledError()])
    mouth3 = FakeMouth()
    cancelled = False
    try:
        with mock.patch.object(btmain, "signals"), \
             mock.patch.object(btmain, "log"):
            asyncio.run(btmain.speak_reply(brain, mouth3, "hi"))
    except asyncio.CancelledError:
        cancelled = True
    check("a cancelled turn propagates", cancelled)
    check("and the brain is interrupted to stop the model",
          brain.interrupted)

    brain = FakeBrain(["One.", "Two."])
    mouth4 = FakeMouth()

    def boom(_s):
        raise RuntimeError("journal broke")

    with mock.patch.object(btmain, "signals"), \
         mock.patch.object(btmain, "log"):
        asyncio.run(btmain.speak_reply(brain, mouth4, "hi", on_reply=boom))
    check("a raising on_reply callback is swallowed",
          mouth4.chunks == ["One.", "Two."], mouth4.chunks)


def permission_gate():
    print("\n--- make_permission_gate: speak, wait, decide ---")
    mouth = FakeMouth()
    mouth.speaking = False

    def reset():
        btmain._PERM["fut"] = None
        btmain._PERM["hinted"] = False
        btmain._AUTOAPPROVE["on"] = False

    def run_gate(tool, tool_input, answers, ctx=None, timeout=8.0):
        async def runner():
            gate = btmain.make_permission_gate(mouth)
            task = asyncio.ensure_future(gate(tool, tool_input, ctx))
            last = None
            for ans in answers:
                for _ in range(4000):
                    fut = btmain._PERM["fut"]
                    if fut is not None and fut is not last and not fut.done():
                        break
                    await asyncio.sleep(0.005)
                last = btmain._PERM["fut"]
                if last is not None and not last.done():
                    last.set_result(ans)
            return await asyncio.wait_for(task, timeout)
        return asyncio.run(runner())

    reset()
    with mock.patch.object(btmain, "signals"), \
         mock.patch.object(btmain, "log"):
        btmain._AUTOAPPROVE["on"] = True
        res = run_gate("Bash", {"command": "ls"}, [])
    check("auto-approve returns allow without asking",
          res.behavior == "allow", res)
    reset()

    with mock.patch.object(btmain, "signals"), \
         mock.patch.object(btmain, "log"):
        res = run_gate("Bash", {"command": "ls"}, ["yes please"])
    check("an affirmative answer allows", res.behavior == "allow", res)
    reset()

    with mock.patch.object(btmain, "signals"), \
         mock.patch.object(btmain, "log"):
        res = run_gate("Bash", {"command": "rm -rf /"}, ["absolutely not"])
    check("a non-affirmative answer denies", res.behavior == "deny", res)
    check("the denial quotes the person",
          "absolutely not" in res.message, res.message)
    reset()

    with mock.patch.object(btmain, "signals"), \
         mock.patch.object(btmain, "log"):
        res = run_gate("Write", {"file_path": "x.md"},
                       [btmain._INTERRUPT_ANSWER])
    check("the interrupt sentinel denies silently",
          res.behavior == "deny" and "interrupted" in res.message.lower(),
          res.message)
    reset()

    with mock.patch.object(btmain, "signals"), \
         mock.patch.object(btmain, "log"):
        res = run_gate("WebFetch", {"url": "https://x"}, ["details", "yes"])
    check("asking for details then yes still allows",
          res.behavior == "allow", res)
    check("details were read back out",
          any("details" in s.lower() for s in mouth.said), mouth.said)
    reset()

    with mock.patch.object(btmain, "signals"), \
         mock.patch.object(btmain, "log"), \
         mock.patch.object(btmain, "PERM_TIMEOUT_S", 0.05):
        res = run_gate("Bash", {"command": "ls"}, [], timeout=5.0)
    check("silence times out into a deny", res.behavior == "deny", res)
    check("the timeout is announced", any("No answer" in s
                                          for s in mouth.said), mouth.said)
    reset()


def gate_helpers():
    print("\n--- _deny_pending resolves a live ask ---")
    async def runner():
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        btmain._PERM["fut"] = fut
        btmain._deny_pending()
        return await fut

    got = asyncio.run(runner())
    check("a pending future is resolved to the interrupt sentinel",
          got == btmain._INTERRUPT_ANSWER, got)
    btmain._PERM["fut"] = None
    btmain._deny_pending()  # no future: must not raise
    check("no future -> no-op", True)


print("=" * 66)
print("MAIN HELPERS, REPLY BATCHING, PERMISSION GATE (offline)")
print("=" * 66)
console()
formatting()
typed_input()
human_forms()
config_write()
speak_first_alone()
speak_empty_and_errors()
permission_gate()
gate_helpers()

print("\n" + "=" * 66)
print("MAIN OK" if not FAILURES else f"MAIN FAILURES: {FAILURES}")
sys.exit(1 if FAILURES else 0)
