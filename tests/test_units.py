# Hardware-free unit tests: pure helpers extracted from echo.py by AST.
# Run: python tests/test_units.py   (needs numpy; scipy optional)
# Extract pure functions from echo.py via AST and test them without audio hardware.
import ast, asyncio, json, math, os, queue, sys, time, tempfile, pathlib
from datetime import datetime, timedelta
import numpy as np
from scipy.signal import resample_poly

SRC = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parent.parent / "echo.py"
src = SRC.read_text()
tree = ast.parse(src)
want = {"resample_int16", "format_duration", "env_flag", "queue_get",
        "save_timers", "register_timer", "restore_timers", "timer_worker",
        "parse_alarm_datetime", "local_timezone", "play_timer_alarm"}
nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in want]
ns = dict(np=np, resample_poly=resample_poly, math=math, os=os, asyncio=asyncio, queue=queue,
          time=time, json=json, datetime=datetime, timedelta=timedelta,
          TIMER_LATE_GRACE_SECONDS=600, timers={}, next_timer_id=1)
tmp = pathlib.Path(tempfile.mkdtemp())
ns["TIMERS_FILE"] = tmp / "timers.json"
fired = []
ns["print"] = lambda *a, **k: None
exec(compile(ast.Module(body=nodes, type_ignores=[]), "echo_extract", "exec"), ns)
# stub out the beep player
async def fake_play(timer_id, label): fired.append((timer_id, label))
ns["play_timer_alarm"] = fake_play

# --- resample_int16
r = ns["resample_int16"]
t = np.arange(16000) / 16000
sig = (np.sin(2*np.pi*440*t) * 10000).astype(np.int16)
out = r(sig, 16000, 24000); assert out.dtype == np.int16 and abs(len(out) - 24000) <= 1, len(out)
out48 = r(out, 24000, 48000); assert abs(len(out48) - 48000) <= 2
ref = r(out, 24000, 16000); assert abs(len(ref) - 16000) <= 2
# round-trip amplitude preserved within a few percent
assert 0.95 < np.abs(ref.astype(float)).max() / 10000 < 1.05
assert len(r(np.array([], dtype=np.int16), 16000, 24000)) == 0
assert r(sig, 16000, 16000) is sig
# clipping doesn't overflow
loud = np.full(1600, 32767, dtype=np.int16); assert r(loud, 16000, 24000).max() <= 32767
print("resample ok")

# --- format_duration / env_flag
f = ns["format_duration"]
assert f(0) == "0 seconds" and f(61) == "1 minute 1 second" and f(3600) == "1 hour"
os.environ["X_FLAG"] = "yes"; assert ns["env_flag"]("X_FLAG") is True
os.environ["X_FLAG"] = "0"; assert ns["env_flag"]("X_FLAG") is False
assert ns["env_flag"]("X_MISSING") is False
print("format/env ok")

# --- queue_get bounded blocking
async def t_queue():
    q = queue.Queue()
    task = asyncio.create_task(ns["queue_get"](q))
    await asyncio.sleep(0.1); task.cancel()
    # the worker thread may hold the queue for up to its 0.5 s poll timeout
    try: await task
    except asyncio.CancelledError: pass
    await asyncio.sleep(0.6)
    q.put("x"); assert await asyncio.wait_for(ns["queue_get"](q), 2) == "x"
asyncio.run(t_queue()); print("queue_get ok")

# --- timer persistence: register, save, restore (future, late-but-in-grace, stale)
async def t_timers():
    ns["timers"].clear()
    ns["register_timer"](1, "pasta", 600, 600)
    data = json.loads(ns["TIMERS_FILE"].read_text())
    assert data[0]["timer_id"] == 1 and data[0]["label"] == "pasta"
    due = datetime.fromisoformat(data[0]["due_at"])
    assert 590 < (due - datetime.now().astimezone()).total_seconds() <= 600
    for tm in ns["timers"].values(): tm["task"].cancel()
    ns["timers"].clear()
    # simulate a restart: write one future, one 30s-late, one 2h-late timer
    now = datetime.now().astimezone()
    ns["TIMERS_FILE"].write_text(json.dumps([
        {"timer_id": 3, "label": "future", "duration_seconds": 100, "due_at": (now + timedelta(seconds=100)).isoformat()},
        {"timer_id": 5, "label": "late", "duration_seconds": 100, "due_at": (now - timedelta(seconds=30)).isoformat()},
        {"timer_id": 7, "label": "stale", "duration_seconds": 100, "due_at": (now - timedelta(hours=2)).isoformat()},
    ]))
    await ns["restore_timers"]()
    assert set(ns["timers"]) == {3, 5}, set(ns["timers"])
    assert ns["next_timer_id"] == 8, ns["next_timer_id"]
    await asyncio.sleep(0.2)          # late timer fires immediately
    assert fired == [(5, "late")], fired
    assert set(ns["timers"]) == {3}
    saved = json.loads(ns["TIMERS_FILE"].read_text()); assert [e["timer_id"] for e in saved] == [3]
    for tm in ns["timers"].values(): tm["task"].cancel()
asyncio.run(t_timers()); print("timers ok")
print("ALL OK")
