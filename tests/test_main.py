"""main.py offline: the pure helpers, the reply batching, the spoken
permission gate, and the whole `amain()` live loop driven over fakes --
pinned without a mic, a model, or an opencode server.
"""
from pathlib import Path
import asyncio
import io
import os
import queue
import sys
import tempfile
import threading
import time
import types
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

    # blank lines around and between pastes: skipped, not emitted.
    r5, w5 = os.pipe()
    on = btmain._PASTE_ON.encode()
    off = btmain._PASTE_OFF.encode()
    os.write(w5, b"line1\n\n" + on + off + b"\nlast\n" + on + b"body" + off
             + b"\n\n")
    os.close(w5)
    q5: queue.Queue = queue.Queue()
    btmain._typed_reader_pipe(q5, r5)
    os.close(r5)
    mixed = []
    while not q5.empty():
        mixed.append(q5.get_nowait())
    check("blank lines are skipped around pastes",
          mixed == ["line1", "last", "body"], mixed)

    import builtins
    q3: queue.Queue = queue.Queue()
    with mock.patch.object(builtins, "input",
                           side_effect=["  hello  ", "", "│> quoted", EOFError]):
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

    # A speakable-only sentence (all backticks / whitespace) is dropped.
    brain = FakeBrain(["```", "Real words."])
    mouth5 = FakeMouth()
    with mock.patch.object(btmain, "signals"), \
         mock.patch.object(btmain, "log"):
        asyncio.run(btmain.speak_reply(brain, mouth5, "hi"))
    check("an unspeakable sentence is dropped, the next one speaks",
          mouth5.chunks == ["Real words."], mouth5.chunks)

    # A raise from on_reply on the batched (second) chunk is swallowed.
    brain = FakeBrain(["One.", "Two.", "Three."])
    mouth6 = FakeMouth()
    with mock.patch.object(btmain, "signals"), \
         mock.patch.object(btmain, "log"):
        asyncio.run(btmain.speak_reply(brain, mouth6, "hi", on_reply=boom))
    check("a raising on_reply mid-batch is swallowed",
          mouth6.chunks == ["One.", "Two. Three."], mouth6.chunks)

    class DeafBrain(FakeBrain):
        async def interrupt(self):
            raise RuntimeError("interrupt broke")

    brain = DeafBrain(["One.", asyncio.CancelledError()])
    mouth7 = FakeMouth()
    cancelled = False
    try:
        with mock.patch.object(btmain, "signals"), \
             mock.patch.object(btmain, "log"):
            asyncio.run(btmain.speak_reply(brain, mouth7, "hi"))
    except asyncio.CancelledError:
        cancelled = True
    check("a failing interrupt still lets the cancel through", cancelled)

    # A turn that dies with one sentence still batched, and an on_reply
    # that also raises: the tail is spoken, the callback's error is eaten.
    brain = FakeBrain(["One.", "Two.", RuntimeError("mid-stream")])
    mouth8 = FakeMouth()
    with mock.patch.object(btmain, "signals"), \
         mock.patch.object(btmain, "log"):
        asyncio.run(btmain.speak_reply(brain, mouth8, "hi", on_reply=boom))
    check("a failing turn still speaks its batched tail",
          mouth8.chunks == ["One.", "Two."], mouth8.chunks)

    # A clean turn that ends holding exactly one extra sentence.
    brain = FakeBrain(["One.", "Two."])
    mouth9 = FakeMouth()
    with mock.patch.object(btmain, "signals"), \
         mock.patch.object(btmain, "log"):
        asyncio.run(btmain.speak_reply(brain, mouth9, "hi"))
    check("a clean turn flushes its single trailing sentence",
          mouth9.chunks == ["One.", "Two."], mouth9.chunks)


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

    with mock.patch.object(btmain, "signals"), \
         mock.patch.object(btmain, "log"):
        run_gate("Bash", {"command": "ls"}, ["yes"])
        res = run_gate("Bash", {"command": "ls"}, ["yes"])
    check("a later ask does not re-announce the escape hatch",
          res.behavior == "allow" and btmain._PERM["hinted"], res)
    reset()

    mouth.speaking = False
    real_wait_for = btmain.asyncio.wait_for
    ticks = {"n": 0}

    async def flaky_wait_for(aw, t):
        if not isinstance(aw, asyncio.Task) and ticks["n"] == 0:
            ticks["n"] = 1
            aw.cancel()
            raise asyncio.TimeoutError
        return await real_wait_for(aw, t)

    with mock.patch.object(btmain, "signals") as sig, \
         mock.patch.object(btmain, "log"), \
         mock.patch.object(btmain.asyncio, "wait_for", flaky_wait_for):
        res = run_gate("Bash", {"command": "ls"}, ["yes"])
    check("a quiet wait tick nudges the state to listening",
          sig.set_state.called and res.behavior == "allow", res)
    reset()

    mouth.speaking = True
    ticks["n"] = 0
    with mock.patch.object(btmain, "signals") as sig2, \
         mock.patch.object(btmain, "log"), \
         mock.patch.object(btmain.asyncio, "wait_for", flaky_wait_for):
        res = run_gate("Bash", {"command": "ls"}, ["yes"])
    check("a wait tick while speaking does not touch the state",
          ("listening",) not in [c.args
                                 for c in sig2.set_state.call_args_list]
          and res.behavior == "allow", res)
    mouth.speaking = False
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


def journal_flush_case():
    print("\n--- journal_flush: a hangup never hangs or crashes ---")

    class FakeJournal:
        def __init__(self, active=True, events=True):
            self.active = active
            self.events = events
            self.written = []

        def transcript(self):
            return "**you:** hi"

        def write(self, summary):
            self.written.append(summary)
            return None

    class FakeScratch:
        instance = None

        def __init__(self, *a, **k):
            FakeScratch.instance = self
            self.started = False
            self.stopped = False

        async def start(self):
            self.started = True

        async def stop(self):
            self.stopped = True

        async def ask_stream(self, text):
            yield "x"

    j = FakeJournal(active=False)
    asyncio.run(btmain.journal_flush(j, None))
    check("an inactive journal is skipped", j.written == [])

    j = FakeJournal(active=True, events=False)
    asyncio.run(btmain.journal_flush(j, None))
    check("an empty journal is skipped", j.written == [])

    j = FakeJournal()
    with mock.patch.object(btmain, "CFG", {"journal_summary": False}), \
         mock.patch.object(btmain, "log"):
        asyncio.run(btmain.journal_flush(j, None))
    check("no summary -> an entry with no summary", j.written == [""])

    async def fake_summarize(ask_stream, text):
        return "the summary"

    j = FakeJournal()
    with mock.patch.object(btmain, "CFG", {"journal_summary": True}), \
         mock.patch.object(btmain, "WarmBrain", FakeScratch), \
         mock.patch.object(btmain, "journal_summarize", fake_summarize), \
         mock.patch.object(btmain, "log"):
        asyncio.run(btmain.journal_flush(j, None))
    check("a summary is written when asked", j.written == ["the summary"])
    check("the scratch brain is started and stopped",
          FakeScratch.instance.started and FakeScratch.instance.stopped)

    class StubbornScratch(FakeScratch):
        async def stop(self):
            raise RuntimeError("stop broke")

    j = FakeJournal()
    with mock.patch.object(btmain, "CFG", {"journal_summary": True}), \
         mock.patch.object(btmain, "WarmBrain", StubbornScratch), \
         mock.patch.object(btmain, "journal_summarize", fake_summarize), \
         mock.patch.object(btmain, "log"):
        asyncio.run(btmain.journal_flush(j, None))
    check("a scratch brain that will not stop is swallowed",
          j.written == ["the summary"])

    class WrittenJournal(FakeJournal):
        def write(self, summary):
            self.written.append(summary)
            return type("P", (), {"name": "entry.md"})()

    j = WrittenJournal()
    with mock.patch.object(btmain, "CFG", {"journal_summary": False}), \
         mock.patch.object(btmain, "log") as logged:
        asyncio.run(btmain.journal_flush(j, None))
    check("a written entry is logged by name", logged.called)

    class BoomJournal(FakeJournal):
        def write(self, summary):  # noqa: ARG002
            raise RuntimeError("disk full")

    j = BoomJournal()
    with mock.patch.object(btmain, "CFG", {"journal_summary": False}), \
         mock.patch.object(btmain, "log"):
        asyncio.run(btmain.journal_flush(j, None))
    check("a broken write never crashes the hangup", True)


def single_instance():
    print("\n--- _claim_single_instance: one voice line, out loud ---")
    taken = mock.Mock()
    taken.bind.side_effect = OSError("in use")
    with mock.patch.object(btmain.socket, "socket", return_value=taken):
        btmain._instance_lock = None
        ok = btmain._claim_single_instance()
    check("a held port refuses a second voice line", ok is False)
    check("the loser socket is closed", taken.close.called)

    free = mock.Mock()
    with mock.patch.object(btmain.socket, "socket", return_value=free):
        btmain._instance_lock = None
        ok = btmain._claim_single_instance()
    check("a free port is claimed",
          ok is True and btmain._instance_lock is free)
    check("the winner listens but serves nothing", free.listen.called)
    btmain._instance_lock = None

    with mock.patch.object(btmain, "signals") as sig:
        btmain._take_the_bus()
    check("taking the bus parks it and registers the park",
          sig.register_exit_park.called and sig.park.called)


class _LoopMouth:
    def __init__(self):
        self.said = []
        self.chunks = []
        self.speaking = False
        self.shut = 0
        self.shut_down = False
        self.waited = 0
        self.ducker = mock.Mock()

    def say(self, s):
        self.said.append(s)

    def say_chunk(self, s, pending=None):
        self.chunks.append(s)

    def shut_up(self):
        self.shut += 1

    def wait_done(self, timeout=None):
        self.waited += 1

    def shutdown(self):
        self.shut_down = True


class _LoopEars:
    def __init__(self, fail_first=0):
        self.calls = 0
        self.fail_first = fail_first

    def listen_once(self, gate=None, abort=None):
        self.calls += 1
        time.sleep(0.01)
        if self.calls <= self.fail_first:
            raise RuntimeError("mic gone")
        return None

    def wait_for_wake(self, phrases, gate=None, abort=None):
        return None


class _LoopJournal:
    def __init__(self, *a, **k):
        self.active = False
        self.events = []

    def say_user(self, t):
        pass

    def say_agent(self, t):
        pass


class _LoopBrain:
    instances = []

    def __init__(self, *a, **k):
        self.model = k.get("model") or "fake-model"
        self.session = {"turns": 0, "out_tokens": 0,
                        "in_tokens": 0, "cost": 0.0}
        self.started = False
        self.stopped = False
        self.commands = []
        self.got = []
        _LoopBrain.instances.append(self)

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    async def reset_turn(self):
        pass

    async def interrupt(self):
        pass

    async def set_permission_mode(self, m):
        pass

    async def command(self, c):
        self.commands.append(c)
        return f"ok {c}"

    async def context_usage(self):
        return {"categories": []}

    async def ask_stream(self, text):
        if text.startswith("Warmup ping"):
            return
        self.got.append(text)
        yield "Answer one."
        yield "Answer two."


class _DeadBrain(_LoopBrain):
    async def start(self):
        raise RuntimeError("no brain, no provider")


class _SlowBrain(_LoopBrain):
    async def ask_stream(self, text):
        if text.startswith("Warmup ping"):
            return
        await asyncio.sleep(0.4)
        yield "Slow answer."


class _PermBrain(_LoopBrain):
    """Mimics the SDK: a turn poses a spoken ask and resumes when the
    NEXT utterance resolves it."""

    async def ask_stream(self, text):
        if text.startswith("Warmup ping"):
            return
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        btmain._PERM["fut"] = fut
        btmain._PERM["asked_at"] = time.monotonic()
        try:
            answer = await asyncio.wait_for(fut, timeout=5)
        finally:
            btmain._PERM["fut"] = None
        yield f"answered {answer}"


class _EdgeBrain(_LoopBrain):
    async def command(self, c):
        self.commands.append(c)
        if c == "/compact":
            raise RuntimeError("compact blew up")
        if c == "/clear":
            return "Error: could not clear the context"
        return f"ok {c}"

    async def set_permission_mode(self, m):
        raise RuntimeError("cannot flip into ask")


class _WarmTokenBrain(_LoopBrain):
    """Yields a token for the hidden warmup ping instead of nothing."""

    async def ask_stream(self, text):
        if text.startswith("Warmup ping"):
            yield "ready"
            return
        self.got.append(text)
        yield "Answer one."


class _WakeEars(_LoopEars):
    """One wake phrase per run, then silence; optional command turn."""

    def __init__(self, wake, then="what is up"):
        super().__init__()
        self._wake = wake
        self._then = then
        self._n = 0

    def wait_for_wake(self, phrases, gate=None, abort=None):
        self._n += 1
        return self._wake if self._n == 1 else None

    def listen_once(self, gate=None, abort=None):
        self.calls += 1
        return self._then if self._n == 1 else None


class _QuitEars(_LoopEars):
    """Returns a quit phrase once, then silence."""

    def __init__(self, text="goodbye seyon"):
        super().__init__()
        self._text = text
        self._said = False

    def listen_once(self, gate=None, abort=None):
        self.calls += 1
        if not self._said:
            self._said = True
            return self._text
        return None


class _BadDeviceEars(_LoopEars):
    """Every capture fails with a device-level error."""

    def listen_once(self, gate=None, abort=None):
        self.calls += 1
        raise RuntimeError("invalid device")


class _FakePTT:
    def __init__(self):
        self.presses = 0

    def wait_press(self):
        self.presses += 1
        if self.presses > 1:
            time.sleep(0.25)
        return None

    def is_held(self):
        return False


def _noop_record(is_held, on_release=None):
    if on_release:
        on_release()
    return ""


def _run_amain(feed, cfg_extra=None, argv=(), brain_cls=_LoopBrain,
               ears=None, mouth=None, ptt=None, record=None,
               feed_delay=0.0, explain=None):
    cfg = {
        "agent_dir": "C:/agent", "model": "fast-model",
        "deep_model": "deep-model", "greeting": "Hello.",
        "signoff": "Bye.", "ptt_key": "home", "mic_mode": "open",
        "permission_mode": "ask", "effort": "", "journal_dir": None,
        "resume_last_session": False, "wake_word": False,
        "wake_word_phrases": [], "barge_in": False,
        "journal_summary": False,
    }
    cfg.update(cfg_extra or {})
    mouth = mouth or _LoopMouth()
    ears = ears or _LoopEars()

    def feeder(q):
        if feed_delay:
            time.sleep(feed_delay)
        for line in feed:
            q.put(line)

    ptt_patch = (mock.patch.object(btmain, "PTTListener",
                                   side_effect=RuntimeError("no key hook"))
                 if ptt is None
                 else mock.patch.object(btmain, "PTTListener",
                                        lambda key: ptt))
    record_patch = (mock.patch.object(btmain, "record_held")
                    if record is None else
                    mock.patch.object(btmain, "record_held", record))

    _LoopBrain.instances.clear()
    btmain._PERM.update(fut=None, hinted=False, asked_at=0.0)
    btmain._CONFIRM.update(verb=None, at=0.0)
    btmain._AUTOAPPROVE["on"] = False
    btmain._MIC.update(mode="ptt", gen=0, btn=False)

    async def runner():
        try:
            async with asyncio.timeout(20):
                await btmain.amain()
        except SystemExit as e:
            result["exit"] = e.code
        except asyncio.TimeoutError:
            result["timeout"] = True

    result = {"exit": None, "timeout": False}
    with mock.patch.object(btmain, "CFG", cfg), \
         mock.patch.object(btmain, "Mouth", lambda: mouth), \
         mock.patch.object(btmain, "Ears", lambda: ears), \
         mock.patch.object(btmain, "WarmBrain", brain_cls), \
         mock.patch.object(btmain, "Journal", _LoopJournal), \
         mock.patch.object(btmain, "make_permission_gate",
                           lambda m: (lambda *a, **k: None)), \
         mock.patch.object(btmain, "_typed_reader", feeder), \
         mock.patch.object(btmain, "set_turn_language"), \
         mock.patch.object(btmain, "warm_ears"), \
         mock.patch.object(btmain, "explain_audio_failure",
                           explain or (lambda e: False)), \
         mock.patch.object(btmain, "_write_config_key",
                           lambda k, v: True), \
         mock.patch.object(btmain, "QUIT_PHRASES", {"goodbye seyon"}), \
         mock.patch.object(btmain, "signals"), \
         mock.patch.object(btmain, "log"), \
         ptt_patch, record_patch, \
         mock.patch.object(btmain.sys, "argv", ["prog", *argv]):
        asyncio.run(runner())
    return mouth, ears, result


def amain_loop():
    print("\n--- amain: the live loop over fakes ---")
    feed = [
        "what is the capital of France",
        "clear the session",
        "compact the session",
        "switch to the deep model",
        "back to the fast model",
        "set effort to low",
        "usage report",
        "go hands free",
        "push to talk mode",
        "stop asking for permission",
        "confirm",
        "start asking again",
        "goodbye seyon",
    ]
    mouth, _, res = _run_amain(feed, argv=["--model"])
    brain = _LoopBrain.instances[-1]
    check("the loop ran to a clean hangup",
          res["exit"] is None and not res["timeout"], res)
    check("the greeting is spoken", "Hello." in mouth.said, mouth.said[:2])
    check("a typed question is answered aloud",
          "Answer one." in mouth.chunks, mouth.chunks)
    check("the signoff is spoken on quit", "Bye." in mouth.said)
    check("the mouth is shut down on the way out", mouth.shut_down)
    check("the brain is stopped on the way out", brain.stopped)
    check("console clear reached the brain",
          "/clear" in brain.commands, brain.commands)
    check("console compact reached the brain", "/compact" in brain.commands)
    check("console deep selected the deep model",
          any("deep-model" in c for c in brain.commands), brain.commands)
    check("console fast selected the fast model",
          any("fast-model" in c for c in brain.commands), brain.commands)
    check("console effort set a level",
          any("effort low" in c for c in brain.commands), brain.commands)
    check("usage was spoken",
          any("this session" in s for s in mouth.said), mouth.said)
    check("the mic was switched to push-to-talk",
          btmain._MIC["mode"] == "ptt", btmain._MIC["mode"])

    mouth2, _, res2 = _run_amain([], brain_cls=_DeadBrain)
    check("a brain that will not connect exits the line",
          res2["exit"] == 1 and not res2["timeout"], res2)
    check("and says so out loud",
          any("brain" in s.lower() for s in mouth2.said), mouth2.said)

    mouth3, _, res3 = _run_amain(
        ["goodbye seyon"],
        cfg_extra={"wake_word": True, "wake_word_phrases": [],
                   "resume_last_session": True},
        argv=["--wake-word"])
    check("wake word with no phrases falls back quietly",
          res3["exit"] is None and not res3["timeout"], res3)
    check("the fallback still hangs up on the phrase",
          "Bye." in mouth3.said, mouth3.said)

    print("\n--- amain: console edges, interrupts, the mic, and the key ---")
    edge_feed = [
        "clear the session",
        "compact the session",
        "stop asking for permission",
        "later on",
        "stop asking for permission",
        "confirm",
        "push to talk mode",
        "go hands free",
        "start asking again",
        "goodbye seyon",
    ]
    mouth4, _, res4 = _run_amain(
        edge_feed, argv=["--model", "alt-model"],
        cfg_extra={"permission_mode": "bypassPermissions",
                   "mic_mode": "ptt", "effort": "bogus"},
        brain_cls=_EdgeBrain)
    check("the edge console run hangs up cleanly",
          res4["exit"] is None and not res4["timeout"], res4)
    check("a console verb that errors is reported, not fatal",
          any("That command hit an error" in s for s in mouth4.said),
          mouth4.said)
    check("a slash-command error is read back verbatim",
          any(s.startswith("Error:") for s in mouth4.said), mouth4.said)
    check("an unanswered confirm stays put",
          any("Staying as we are" in s for s in mouth4.said), mouth4.said)
    check("with no key hook, the boot ptt falls back to the open mic",
          any("Push to talk. Hold the home key" in s for s in mouth4.said),
          mouth4.said)
    check("hands-free can be turned on live",
          btmain._MIC["mode"] == "open", btmain._MIC["mode"])
    check("a failed live flip into 'ask' is admitted",
          any("couldn't switch over" in s for s in mouth4.said), mouth4.said)

    mouth5, _, res5 = _run_amain(["ask one", "ask two", "goodbye seyon"],
                                 brain_cls=_SlowBrain)
    check("a mid-reply interruption hangs up cleanly",
          res5["exit"] is None and not res5["timeout"], res5)
    check("the running reply was cut at least once", mouth5.shut >= 2,
          mouth5.shut)

    mouth6, _, res6 = _run_amain(
        ["do thing one", "yes please", "do thing two", "goodbye seyon"],
        brain_cls=_PermBrain)
    check("the permission run hangs up cleanly",
          res5["exit"] is None and not res5["timeout"], res5)
    check("a spoken answer resumes the paused turn",
          any("answered yes please" in c for c in mouth6.chunks),
          mouth5.chunks)

    def make_rec():
        state = {"n": 0}

        def rec(is_held, on_release=None):
            state["n"] += 1
            if on_release:
                on_release()
            return "hello via ptt" if state["n"] == 1 else ""
        return rec

    mouth6b, _, res6b = _run_amain(
        ["goodbye seyon"], cfg_extra={"mic_mode": "ptt"},
        ptt=_FakePTT(), record=make_rec(), feed_delay=0.5)
    check("the push-to-talk run hangs up cleanly",
          res6b["exit"] is None and not res6b["timeout"], res6b)
    check("a held key sends the recorded words to the brain",
          any("hello via ptt" in b.got for b in _LoopBrain.instances),
          [b.got for b in _LoopBrain.instances])

    mouth6c, _, res6c = _run_amain(
        ["push to talk mode", "goodbye seyon"],
        cfg_extra={"mic_mode": "ptt"}, ptt=_FakePTT(),
        record=_noop_record)
    check("push-to-talk already on says so",
          any("Already on push to talk" in s for s in mouth6c.said),
          mouth6c.said)

    mouth7, _, res7 = _run_amain(
        ["goodbye seyon"], cfg_extra={"mic_mode": "open"},
        ears=_LoopEars(), feed_delay=0.3)
    check("the open-mic run hangs up cleanly (silent captures)",
          res7["exit"] is None and not res7["timeout"], res7)
    check("the open mic was actually polled",
          mouth7 is not None, mouth7)

    mouth8, _, res8 = _run_amain(
        ["goodbye seyon"], cfg_extra={"mic_mode": "open"},
        ears=_LoopEars(fail_first=3), feed_delay=0.5)
    check("the open-mic failure run hangs up cleanly",
          res8["exit"] is None and not res8["timeout"], res8)
    check("three mic failures fall back to push-to-talk",
          any("open microphone keeps failing" in s for s in mouth8.said),
          mouth8.said)

    print("\n--- amain: startup config and the capture edges ---")
    mouth9, _, res9 = _run_amain(["goodbye seyon"],
                                 brain_cls=_WarmTokenBrain)
    check("a warmup reply is consumed silently",
          res9["exit"] is None and not res9["timeout"], res9)

    mouth10, _, res10 = _run_amain(["goodbye seyon"],
                                   cfg_extra={"effort": "high"})
    check("a configured effort is applied at boot",
          res9["exit"] is None and not res9["timeout"], res9)
    check("the boot effort reached the brain",
          any("effort high" in c for c in _LoopBrain.instances[-1].commands),
          _LoopBrain.instances[-1].commands)

    with tempfile.TemporaryDirectory() as td:
        sess = Path(td) / "session.json"
        sess.write_text("saved-session-id\n", encoding="utf-8")
        import backtalk.brain as btbrain
        with mock.patch.object(btbrain, "SESSION_FILE", sess):
            _, _, res10 = _run_amain(
                ["goodbye seyon"],
                cfg_extra={"resume_last_session": True})
    check("a saved session file is resumed",
          res9["exit"] is None and not res9["timeout"], res9)

    _, _, res_w1 = _run_amain(
        ["goodbye seyon"],
        cfg_extra={"mic_mode": "open", "wake_word": True,
                   "wake_word_phrases": ["vijay seyon"]},
        ears=_WakeEars(wake="vijay seyon hello"), feed_delay=0.3)
    check("a wake phrase that carried the command is used",
          res_w1["exit"] is None and not res_w1["timeout"], res_w1)

    _, _, res_w2 = _run_amain(
        ["goodbye seyon"],
        cfg_extra={"mic_mode": "open", "wake_word": True,
                   "wake_word_phrases": ["vijay seyon"]},
        ears=_WakeEars(wake="vijay seyon"), feed_delay=0.3)
    check("a bare wake phrase then listens for the command",
          res_w2["exit"] is None and not res_w2["timeout"], res_w2)

    _, _, res_bd = _run_amain(
        ["goodbye seyon"], cfg_extra={"mic_mode": "open"},
        ears=_BadDeviceEars(), feed_delay=0.4,
        explain=lambda e: True)
    check("a recognized device failure falls back quiet but clean",
          res_bd["exit"] is None and not res_bd["timeout"], res_bd)

    _, _, res_q = _run_amain(
        ["goodbye seyon"], cfg_extra={"mic_mode": "open"}, ears=_QuitEars(),
        feed_delay=0.3)
    check("a quit phrase captured by the open mic hangs up",
          res_q["exit"] is None and not res_q["timeout"], res_q)

    def rec_boom(is_held, on_release=None):
        if on_release:
            on_release()
        raise RuntimeError("invalid device")

    _, _, res_rb = _run_amain(
        ["goodbye seyon"], cfg_extra={"mic_mode": "ptt"},
        ptt=_FakePTT(), record=rec_boom, feed_delay=0.5,
        explain=lambda e: True)
    check("a device failure during a held key is spoken plainly",
          res_rb["exit"] is None and not res_rb["timeout"], res_rb)

    def rec_boom_hard(is_held, on_release=None):
        if on_release:
            on_release()
        raise RuntimeError("strange")

    _, _, res_rbh = _run_amain(
        ["goodbye seyon"], cfg_extra={"mic_mode": "ptt"},
        ptt=_FakePTT(), record=rec_boom_hard, feed_delay=0.5)
    check("an unrecognized record failure is only logged",
          res_rbh["exit"] is None and not res_rbh["timeout"], res_rbh)

    def rec_quit(is_held, on_release=None):
        if on_release:
            on_release()
        return "goodbye seyon"

    _, _, res_rq = _run_amain(
        ["goodbye seyon"], cfg_extra={"mic_mode": "ptt"},
        ptt=_FakePTT(), record=rec_quit, feed_delay=0.5)
    check("a quit phrase spoken into the key hangs up",
          res_rq["exit"] is None and not res_rq["timeout"], res_rq)

    mouth_cf, _, res_cf = _run_amain(
        ["stop asking for permission", "goodbye seyon"])
    check("a quit phrase cancels a pending confirm and hangs up",
          res_cf["exit"] is None and not res_cf["timeout"], res_cf)
    check("the signoff still plays",
          "Bye." in mouth_cf.said, mouth_cf.said)


def tty_reader():
    print("\n--- _typed_reader: the POSIX line editor ---")
    reads = iter([
        b"Hel",
        b"lo\x7f\x08p",              # backspace twice, then "p" -> "Help"
        b"\x1b[20",                  # partial paste-on marker, held back
        b"0~" + b"X" * 70,           # marker completes, 70-char paste body
        b"\x1b[20",                  # partial paste-off marker, held back
        b"1~",                       # paste closes (70 chars -> a count)
        b"\r",                       # Enter sends the composed line
        b"",                         # EOF ends the reader
    ])

    def fake_read(fd, n):
        return next(reads, b"")

    termios = types.ModuleType("termios")
    termios.TCSADRAIN = 1
    termios.tcgetattr = lambda fd: ["saved"]
    termios.tcsetattr = lambda fd, when, attrs: None
    tty = types.ModuleType("tty")
    tty.setcbreak = lambda fd: None

    out = io.StringIO()
    fake_sys = types.SimpleNamespace(
        stdin=types.SimpleNamespace(fileno=lambda: 0),
        stdout=out)
    q = queue.Queue()
    with mock.patch.dict(sys.modules, {"termios": termios, "tty": tty}), \
         mock.patch.object(btmain.os, "isatty", lambda fd: True), \
         mock.patch.object(btmain.os, "read", fake_read), \
         mock.patch.object(btmain, "sys", fake_sys):
        btmain._typed_reader(q)

    got = []
    while not q.empty():
        got.append(q.get_nowait())
    line = "Help " + "X" * 70
    check("the tty editor assembled the line across reads",
          got == [line], got)
    check("a long paste collapsed to a count",
          "[pasted 70 chars]" in out.getvalue(), out.getvalue()[:80])
    check("backspace erased characters in the echo",
          "\b \b" in out.getvalue(), out.getvalue()[:80])
    check("the reader restored the terminal on the way out",
          "\x1b[?2004l" in out.getvalue(), out.getvalue()[:80])


class _Stop(Exception):
    pass


def tty_reader_branches():
    print("\n--- _typed_reader: the odd branches ---")

    def make_termios(boom=False):
        m = types.ModuleType("termios")
        m.TCSADRAIN = 1
        m.tcgetattr = lambda fd: ["saved"]

        def setter(fd, when, attrs):
            if boom:
                raise OSError("gone")

        m.tcsetattr = setter
        return m

    def make_sys(out):
        return types.SimpleNamespace(
            stdin=types.SimpleNamespace(fileno=lambda: 0),
            stdout=out)

    with mock.patch.dict(sys.modules, {"termios": None, "tty": None}), \
         mock.patch.object(btmain.os, "isatty", lambda fd: True), \
         mock.patch.object(btmain, "_typed_reader_simple") as simple, \
         mock.patch.object(btmain, "sys", make_sys(io.StringIO())):
        btmain._typed_reader(queue.Queue())
    check("a platform without termios uses the simple reader", simple.called)

    out = io.StringIO()
    tty_ok = types.ModuleType("tty")
    tty_ok.setcbreak = lambda fd: None
    with mock.patch.dict(sys.modules, {"termios": make_termios(boom=True),
                                       "tty": tty_ok}), \
         mock.patch.object(btmain.os, "isatty", lambda fd: True), \
         mock.patch.object(btmain.os, "read", side_effect=OSError("gone")), \
         mock.patch.object(btmain, "sys", make_sys(out)):
        btmain._typed_reader(queue.Queue())
    check("a read error restores the terminal and stops",
          "\x1b[?2004l" in out.getvalue(), out.getvalue()[:40])

    reads = iter([
        b"\x01",                    # a control char: ignored
        b"\x7f",                    # backspace with an empty buffer
        b"\r",                      # Enter on an empty buffer: nothing queued
        b"hi ",                     # a line ending in a space
        b"\x1b[200~\x1b[201~",      # an empty paste: nothing to insert
        b"\x1b[200~body\x1b[201~",  # a real paste onto the "hi " buffer
        b"\r",                      # send "hi body"
        b"",                        # EOF
    ])

    def fake_read(fd, n):
        return next(reads, b"")

    out2 = io.StringIO()
    tty = types.ModuleType("tty")
    tty.setcbreak = lambda fd: None
    q = queue.Queue()
    with mock.patch.dict(sys.modules, {"termios": make_termios(), "tty": tty}), \
         mock.patch.object(btmain.os, "isatty", lambda fd: True), \
         mock.patch.object(btmain.os, "read", fake_read), \
         mock.patch.object(btmain, "sys", make_sys(out2)):
        btmain._typed_reader(q)
    got = []
    while not q.empty():
        got.append(q.get_nowait())
    check("the odd branches still compose the right line",
          got == ["hi body"], got)


def main_entry():
    print("\n--- main(): the process boundary ---")

    def run(claim=True, run_exc=None, log_boom=False, park_boom=False):
        rec = {"exit": [], "os_exit": [], "park": 0, "bus": 0, "log": []}

        def fake_run(coro):
            coro.close()
            if run_exc is not None:
                raise run_exc

        def fake_exit(code=0):
            rec["exit"].append(code)
            raise SystemExit(code)

        def fake_os_exit(code):
            rec["os_exit"].append(code)
            raise _Stop

        def fake_log(s):
            if log_boom:
                raise BaseException("the log itself died")
            rec["log"].append(s)

        def fake_park():
            if park_boom:
                raise BaseException("park died")
            rec["park"] += 1

        sig = types.SimpleNamespace(park=fake_park)
        with mock.patch.object(btmain, "_claim_single_instance",
                               lambda: claim), \
             mock.patch.object(btmain, "_take_the_bus",
                               lambda: rec.__setitem__("bus",
                                                       rec["bus"] + 1)), \
             mock.patch.object(btmain.asyncio, "run", fake_run), \
             mock.patch.object(btmain, "signals", sig), \
             mock.patch.object(btmain, "log", fake_log), \
             mock.patch.object(btmain.sys, "exit", fake_exit), \
             mock.patch.object(btmain.os, "_exit", fake_os_exit):
            try:
                btmain.main()
            except (SystemExit, _Stop):
                pass
        return rec

    rec = run(claim=False)
    check("a second voice line refuses to start",
          rec["exit"] == [1] and rec["bus"] == 0 and rec["park"] == 0, rec)

    rec = run()
    check("the clean path parks the bus and hard-exits 0",
          rec["os_exit"] == [0] and rec["park"] == 1 and rec["bus"] == 1,
          rec)

    rec = run(run_exc=KeyboardInterrupt())
    check("an interrupt still parks and hard-exits 0",
          rec["os_exit"] == [0] and rec["park"] == 1, rec)

    rec = run(run_exc=RuntimeError("kaboom"))
    check("a crash is logged and hard-exits 1",
          rec["os_exit"] == [1]
          and any("CRASH" in s for s in rec["log"]), rec)

    rec = run(run_exc=RuntimeError("kaboom"), log_boom=True)
    check("a crash path survives the log itself failing",
          rec["os_exit"] == [1], rec)

    rec = run(park_boom=True)
    check("the clean path survives a failing park",
          rec["os_exit"] == [0], rec)


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
journal_flush_case()
single_instance()
amain_loop()
tty_reader()
tty_reader_branches()
main_entry()

print("\n" + "=" * 66)
print("MAIN OK" if not FAILURES else f"MAIN FAILURES: {FAILURES}")
sys.exit(1 if FAILURES else 0)
