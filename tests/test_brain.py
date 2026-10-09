"""The brain, offline: every WarmBrain path driven by a fake opencode server.

No model, no network, no 90s timeouts. A FakeServer is swapped in for
brain._Server so the real WarmBrain runs its real code -- session setup,
sentence streaming, reasoning-vs-text filtering, the empty-turn and
timeout guards, session errors, permission routing, the slash-commands,
usage tallying. A second part boots a real localhost HTTP+SSE server so
_Server's own request/health/event-stream code is exercised for real too.

If opencode's wire format changes, this is the test that notices without
needing a model behind it.
"""
import asyncio
import http.server
import json
import queue
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtalk import brain                                  # noqa: E402
from backtalk.config import CFG                             # noqa: E402

CFG["live_data"] = False
CFG["show_usage"] = False
CFG["resume_last_session"] = False

failures = []


def check(name, cond, detail=""):
    print("    %s %s%s" % ("ok  " if cond else "FAIL", name,
                           "" if cond else "   <- " + str(detail)))
    if not cond:
        failures.append(name)


class FakeServer:
    """Stands in for brain._Server: canned replies, scripted events."""

    def __init__(self, *a, **k):
        self.sid = "ses_test"
        self.requests = []
        self.events = []
        self.stopped = False
        self.fail_prompt = False
        self.fail_resume = False

    async def ensure(self):
        pass

    async def stop(self):
        self.stopped = True

    async def request(self, method, path, body=None, directory=None,
                      timeout=300):
        self.requests.append((method, path, body))
        if self.fail_prompt and "prompt_async" in path:
            raise RuntimeError("prompt exploded")
        if self.fail_resume and method == "GET" and path.startswith("/session/ses_old"):
            raise RuntimeError("gone")
        if method == "POST" and path == "/session":
            return {"id": self.sid}
        if method == "GET" and path == f"/session/{self.sid}/message":
            return [{"info": {"role": "assistant", "cost": 0.02,
                              "tokens": {"input": 10, "output": 40}}}]
        if method == "GET" and path.startswith(f"/session/{self.sid}"):
            return {"tokens": {"input": 100, "output": 20, "reasoning": 5,
                               "cache": {"read": 7}}}
        return {}

    async def next_event(self, timeout=None):
        return self.events.pop(0) if self.events else None


async def new_brain(**kw):
    fake = FakeServer()
    saved = brain._Server
    brain._Server = lambda *a, **k: fake
    b = brain.WarmBrain(**kw)
    await b.start()
    return fake, b, saved


async def scenario_stream():
    fake, b, saved = await new_brain()
    try:
        sid = b._sid
        fake.events = [
            {"type": "message.part.updated",
             "properties": {"sessionID": sid,
                            "part": {"id": "p1", "type": "text"}}},
            {"type": "message.part.updated",           # reasoning is not spoken
             "properties": {"sessionID": sid,
                            "part": {"id": "r1", "type": "reasoning"}}},
            {"type": "message.part.delta",
             "properties": {"sessionID": sid, "partID": "r1",
                            "field": "text", "delta": "thinking out loud. "}},
            {"type": "message.part.delta",
             "properties": {"sessionID": "other", "partID": "p1",
                            "field": "text", "delta": "from another session. "}},
            {"type": "message.part.delta",
             "properties": {"sessionID": sid, "partID": "p1",
                            "field": "text", "delta": "Hello there. "}},
            {"type": "message.part.delta",
             "properties": {"sessionID": sid, "partID": "p1",
                            "field": "text", "delta": "How are you today?"}},
            {"type": "session.idle", "properties": {"sessionID": sid}},
        ]
        out = [c async for c in b.ask_stream("hi")]
        check("streamed the spoken sentences",
              out == ["Hello there.", "How are you today?"], out)
        check("never spoke the reasoning part",
              all("thinking" not in c for c in out), out)
        check("ignored another session's delta",
              all("another session" not in c for c in out), out)
        check("session tallied one turn", b.session["turns"] == 1, b.session)
        check("usage tokens recorded",
              b.session["in_tokens"] == 10 and b.session["out_tokens"] == 40,
              b.session)
    finally:
        brain._Server = saved


async def scenario_tool_flush():
    fake, b, saved = await new_brain()
    try:
        sid = b._sid
        fake.events = [
            {"type": "message.part.delta",
             "properties": {"sessionID": sid, "partID": "p1",
                            "field": "text", "delta": "Let me check"}},
            {"type": "message.part.updated",
             "properties": {"sessionID": sid,
                            "part": {"id": "s1", "type": "step-finish",
                                     "reason": "tool-calls"}}},
            {"type": "session.idle", "properties": {"sessionID": sid}},
        ]
        out = [c async for c in b.ask_stream("check something")]
        check("a tool boundary flushes the pending line",
              out == ["Let me check"], out)
    finally:
        brain._Server = saved


async def scenario_empty_and_error():
    fake, b, saved = await new_brain()
    try:
        sid = b._sid
        fake.events = [{"type": "session.idle",
                        "properties": {"sessionID": sid}}]
        out = [c async for c in b.ask_stream("say nothing")]
        check("an empty turn still says something",
              len(out) == 1 and "empty" in out[0], out)

        fake.events = [{"type": "session.error",
                        "properties": {"sessionID": sid, "error": "boom"}}]
        out = [c async for c in b.ask_stream("break")]
        check("a session error is spoken, not swallowed",
              any("did not work" in c for c in out), out)
    finally:
        brain._Server = saved


async def scenario_timeout():
    fake, b, saved = await new_brain()
    old = CFG.get("turn_timeout")
    CFG["turn_timeout"] = 0.0001
    try:
        sid = b._sid
        fake.events = []                       # next_event returns None
        out = [c async for c in b.ask_stream("slow")]
        check("a stalled turn apologises instead of hanging",
              len(out) == 1 and "did not get an answer" in out[0], out)
        check("the stall interrupts the turn",
              any("abort" in p for _, p, _ in fake.requests), fake.requests)
    finally:
        CFG["turn_timeout"] = old
        brain._Server = saved


async def scenario_prompt_failure():
    fake, b, saved = await new_brain()
    try:
        fake.fail_prompt = True
        raised = False
        try:
            async for _ in b.ask_stream("fail"):
                pass
        except RuntimeError:
            raised = True
        check("a prompt failure propagates", raised)
        check("a failed prompt leaves the turn clean", b._dirty is False)
    finally:
        brain._Server = saved


async def scenario_resume():
    fake = FakeServer()
    saved = brain._Server
    brain._Server = lambda *a, **k: fake
    try:
        b = brain.WarmBrain(resume_id="ses_old")
        await b.start()
        check("an existing session is resumed", b._sid == "ses_old", b._sid)
        fake.fail_resume = True
        b2 = brain.WarmBrain(resume_id="ses_old")
        await b2.start()
        check("a lost session falls back to a fresh one",
              b2._sid == "ses_test", b2._sid)
    finally:
        brain._Server = saved


async def scenario_permissions():
    async def allow(*a):
        return _Decision("allow")

    async def deny(*a):
        return _Decision("deny")

    for mode, gate, expect_allow in [
        ("bypassPermissions", None, True),
        ("ask", allow, True),
        ("ask", deny, False),
    ]:
        fake = FakeServer()
        saved = brain._Server
        brain._Server = lambda *a, **k: fake
        try:
            b = brain.WarmBrain(can_use_tool=gate)
            await b.start()
            b._perm_mode = mode
            sid = b._sid
            fake.events = [
                {"type": "permission.asked",
                 "properties": {"sessionID": sid, "id": "per1",
                                "permission": "bash",
                                "patterns": ["rm -rf /"]}},
                {"type": "session.idle", "properties": {"sessionID": sid}},
            ]
            async for _ in b.ask_stream("do it"):
                pass
            replies = [body for _, p, body in fake.requests
                       if "permissions/per1" in p]
            got = replies[-1]["response"] if replies else None
            want = "once" if expect_allow else "reject"
            check("permission %s/%s -> %s" % (mode, bool(gate), want),
                  got == want, got)
        finally:
            brain._Server = saved


class _Decision:
    def __init__(self, behavior):
        self.behavior = behavior


async def scenario_commands():
    fake, b, saved = await new_brain()
    try:
        sid = b._sid
        fake.events = [{"type": "session.idle",
                        "properties": {"sessionID": sid}}]
        check("plain text goes to the model",
              await b.command("just talk") == "ok")

        check("model is reported", (await b.command("/model")).startswith("model is"))
        check("model is switched",
              await b.command("/model new/model") == "model set to new/model")
        check("effort is set",
              await b.command("/effort high") == "effort set to high")
        check("a bad effort is refused",
              "unknown effort" in await b.command("/effort nope"))
        check("compact runs", await b.command("/compact") == "compacted")

        b._dirty = True
        fake.events = [{"type": "session.idle",
                        "properties": {"sessionID": sid}}]
        check("clear restarts the session", await b.command("/clear") == "cleared")
        check("clear stopped the old server", fake.stopped)

        check("a failing command is reported, not raised",
              (await b.command("/effort x")).startswith("unknown effort"))
    finally:
        brain._Server = saved


async def scenario_usage_and_interrupt():
    fake, b, saved = await new_brain()
    try:
        ctx = await b.context_usage()
        check("context usage is summed", ctx["categories"][0]["tokens"] == 132,
              ctx)
        await b.interrupt()
        check("interrupt aborts the session",
              any("abort" in p for _, p, _ in fake.requests), fake.requests)
    finally:
        brain._Server = saved


# ---- the real _Server over a real localhost HTTP + SSE server -----------

class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def _json(self, body):
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.startswith("/global/health"):
            return self._json({"ok": True})
        if self.path.startswith("/event"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            while True:
                item = self.server.sse_queue.get()
                if item is None:
                    return
                if isinstance(item, dict) and "__raw__" in item:
                    line = item["__raw__"] + "\n"
                else:
                    line = "data: %s\n\n" % item
                self.wfile.write(line.encode())
                self.wfile.flush()
        if "/session" in self.path:
            return self._json({"tokens": {"input": 3, "output": 4}})
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n:
            self.rfile.read(n)
        return self._json({})


async def scenario_real_server():
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.sse_queue = queue.Queue()
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    old_port = CFG.get("opencode_port")
    CFG["opencode_port"] = port
    try:
        srv = brain._Server()
        await srv.ensure()                    # health path, no spawn
        check("a healthy server is reused", srv.proc is None)

        got = await srv.request("GET", "/session/abc")
        check("a real request round-trips", got == {"tokens": {"input": 3, "output": 4}}, got)

        q = srv._url("/event", "/tmp/x")
        check("a directory becomes a query", q.endswith("?directory=%2Ftmp%2Fx"), q)

        httpd.sse_queue.put({"__raw__": "this line is not an event"})
        httpd.sse_queue.put("data: {not json")
        httpd.sse_queue.put(json.dumps({"type": "ping", "properties": {"n": 1}}))
        ev = await srv.next_event(timeout=5)
        check("junk lines are skipped, a real event survives",
              ev is not None and ev.get("type") == "ping", ev)

        empty = await srv.next_event(timeout=0.2)
        check("an idle stream times out to None", empty is None, empty)

        httpd.sse_queue.put(None)
        srv._reader.cancel()
    finally:
        CFG["opencode_port"] = old_port
        httpd.shutdown()


async def main():
    await scenario_stream()
    await scenario_tool_flush()
    await scenario_empty_and_error()
    await scenario_timeout()
    await scenario_prompt_failure()
    await scenario_resume()
    await scenario_permissions()
    await scenario_commands()
    await scenario_usage_and_interrupt()
    await scenario_real_server()


print("=" * 66)
print("THE BRAIN, OFFLINE")
print("=" * 66)
asyncio.run(main())
print("\n" + "=" * 66)
print("BRAIN OK" if not failures else "BRAIN FAILURES: %s" % failures)
sys.exit(1 if failures else 0)
