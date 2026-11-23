"""End-to-end test against a running server.  python tests/e2e.py [base_url]"""
import os
import sys
import tempfile
import time

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "sdk"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))

import numpy as np                              # noqa: E402
from qortex import Client, QortexError          # noqa: E402
import sim_robot                                # noqa: E402

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
c = Client.signup("e2e@example.com", URL)
print("key ok")

try:
    Client("qx_bad", URL).get("/v1/robots")
    raise SystemExit("FAIL: bad key accepted")
except QortexError as e:
    assert e.status == 401

assert len(c.models()) >= 8
rows = []
for rtype, compute in [("arm-2dof", "jetson-orin-nano"), ("mobile-base", "raspberry-pi-5"), ("drone", "jetson-orin-nx-16")]:
    t0 = time.time()
    r = c.create_robot(f"test-{rtype}", rtype, compute=compute, power_budget_w=15)
    rec = c.recommend(robot_id=r["id"], task={"kind": "single_task"}, constraints={"deploy": "on_robot", "min_hz": 20})
    top = rec["recommendations"][0]
    assert top["model"] == "qortex/mlp-bc", top
    ds = c.create_dataset(r["id"])
    ds = c.generate(ds["id"], episodes=150)
    assert ds["status"] == "ready" and ds["episodes"] == 150, ds
    teacher = c.finetune(r["id"], ds["id"], base_model=top["model"], epochs=30)
    assert teacher["status"] == "ready" and teacher["latest_eval"], teacher
    student = c.distill(teacher["id"], hidden=(32, 32), quantize="int8", epochs=40)
    assert student["parent_policy_id"] == teacher["id"] and student["latest_eval"]
    te, se = teacher["latest_eval"], student["latest_eval"]

    # gate blocks an impossible bar, then passes a sane one
    try:
        c.deploy(student["id"], r["id"], min_success_rate=1.01)
        raise SystemExit("FAIL: gate did not block")
    except QortexError as e:
        assert e.status == 412, e
    d1 = c.deploy(teacher["id"], r["id"], min_success_rate=0.8)
    d2 = c.deploy(student["id"], r["id"], min_success_rate=0.8)
    assert d2["version"] == 2 and d2["previous_deployment_id"] == d1["id"]

    # the "robot": runtime pulls the artifact, verifies the checksum, runs closed loop
    cache = tempfile.mkdtemp()
    sr, stats, agent = sim_robot.run(c, r["id"], rtype, episodes=20, cache_dir=cache)
    assert agent.deployment["policy_id"] == student["id"]
    tel = c.get(f"/v1/robots/{r['id']}/telemetry")["telemetry"]
    assert tel and tel[0]["successes"] == round(sr * 20)

    # rollback, and the runtime hot-swaps
    c.rollback(r["id"])
    assert agent.sync() and agent.deployment["policy_id"] == teacher["id"]

    # hosted inference matches local inference
    obs = np.zeros(r["obs_dim"]).tolist()
    assert np.allclose(c.act(teacher["id"], obs), agent.policy.act(obs), atol=1e-4)

    # the router now has measured evidence for this account
    rec2 = c.recommend(robot_id=r["id"])
    assert rec2["recommendations"][0]["evidence"] == "measured"

    rows.append((rtype, teacher["n_params"], te["success_rate"], student["n_params"], se["success_rate"],
                 student["size_bytes"], sr, round(time.time() - t0, 1)))
    print("ok", rows[-1])

# custom robot: uploaded data, offline eval, hosted + on-robot path
r = c.create_robot("my-arm", "custom", obs_dim=4, action_dim=2)
rng = np.random.default_rng(0)
eps = []
for _ in range(20):
    o = rng.uniform(-1, 1, (50, 4))
    eps.append({"observations": o.tolist(), "actions": np.tanh(o[:, :2] - o[:, 2:]).tolist()})
ds = c.create_dataset(r["id"])
ds = c.upload_episodes(ds["id"], eps)
assert ds["steps"] == 1000
p = c.finetune(r["id"], ds["id"], epochs=30)
ev = p["latest_eval"]
assert ev["mode"] == "offline" and ev["validated"] and ev["held_out_episodes"] == 4 and ev["scored_steps"] == 200, ev
assert p["training"]["train_steps"] == 800, p["training"]           # the 4 held-out episodes were not trained on
assert ev["action_mse"] < 0.02, ev
st = c.distill(p["id"])
assert st["latest_eval"]["validated"] and st["latest_eval"]["held_out_episodes"] == 4
rec = c.recommend(robot_id=r["id"])
mine = [x for x in rec["recommendations"] if x["model"] == "qortex/mlp-bc"][0]
assert mine["evidence"] == "offline-check" and "not a closed-loop result" in " ".join(mine["why"]), mine

# a single-episode dataset cannot be validated, and the gate refuses it
r1 = c.create_robot("one-ep", "custom", obs_dim=4, action_dim=2)
d1 = c.create_dataset(r1["id"]); c.upload_episodes(d1["id"], eps[:1])
p1 = c.finetune(r1["id"], d1["id"], epochs=5)
assert p1["latest_eval"]["validated"] is False
try:
    c.deploy(p1["id"], r1["id"], max_action_mse=1.0)
    raise SystemExit("FAIL: unvalidated policy deployed")
except QortexError as e:
    assert e.status == 412 and "held-out" in e.message, e

# tenant isolation: a second key cannot read, train on, or deploy the first key's objects
other = Client.signup("other@example.com", URL)
for path in (f"/v1/robots/{r['id']}", f"/v1/datasets/{ds['id']}", f"/v1/policies/{p['id']}", f"/v1/policies/{p['id']}/artifact",
             f"/v1/robots/{r['id']}/deployment", f"/v1/robots/{r['id']}/deployments",
             f"/v1/robots/{r['id']}/telemetry", f"/v1/evals/{p['latest_eval']['id']}", f"/v1/jobs/{p['job_id']}"):
    try:
        other.get(path)
        raise SystemExit(f"FAIL: other key read {path}")
    except QortexError as e:
        assert e.status == 404, (path, e)
for call in (lambda: other.act(p["id"], [0, 0, 0, 0]), lambda: other.deploy(p["id"], r["id"], force=True),
             lambda: other.distill(p["id"], wait=False), lambda: other.rollback(r["id"]),
             lambda: other.finetune(r["id"], ds["id"], wait=False)):
    try:
        call()
        raise SystemExit("FAIL: other key acted on a foreign object")
    except QortexError as e:
        assert e.status == 404, e
assert other.get("/v1/robots")["robots"] == [] and other.get("/v1/policies")["policies"] == []

# validation errors name the field; empty telemetry is rejected
for body, needle in (({}, "name"), ({"name": "x", "type": "nope"}, "type must be")):
    try:
        c.post("/v1/robots", body)
        raise SystemExit("FAIL")
    except QortexError as e:
        assert e.status == 422 and needle in e.message, e
try:
    c.telemetry(r["id"])
    raise SystemExit("FAIL: empty telemetry accepted")
except QortexError as e:
    assert e.status == 422
dep = c.deploy(p["id"], r["id"], max_action_mse=0.02)
assert dep["gate_passed"]
try:
    c.upload_episodes(ds["id"], [{"observations": [[0, 0, 0]], "actions": [[0, 0]]}])
    raise SystemExit("FAIL: bad shape accepted")
except QortexError as e:
    assert e.status == 422
print("ok custom", p["latest_eval"]["action_mse"])

# a VLA base model is accepted but honestly parked
ds2 = c.get(f"/v1/datasets/{ds['id']}")
v = c.finetune(r["id"], ds2["id"], base_model="lerobot/smolvla")
assert v["status"] == "waiting_for_gpu_worker" and "message" in v
try:
    c.finetune(r["id"], ds2["id"], base_model="dream-labs/dream-1")
    raise SystemExit("FAIL")
except QortexError as e:
    assert e.status == 422

# router behaviour
rec = c.recommend(robot={"type": "arm", "compute": "jetson-orin-nx-16"}, task={"kind": "language", "needs_vision": True},
                  constraints={"deploy": "on_robot", "min_hz": 5})
ids = [x["model"] for x in rec["recommendations"]]
assert "qortex/mlp-bc" not in ids and "dream-labs/dream-1" not in ids and ids[0] == "lerobot/smolvla", ids
assert rec["suggested_pipeline"] and rec["pipeline_executable_here"] is False and rec["pipeline_blocked_by"], rec
assert rec["executable_alternative"] is None and rec["executable_alternative_note"]      # nothing runnable does vision + language
ok = c.recommend(robot={"type": "drone", "compute": "jetson-orin-nx-16"})
assert ok["pipeline_executable_here"] is True and ok["pipeline_blocked_by"] == []
spec = c.get("/openapi.json")
art = spec["paths"]["/v1/policies/{id}/artifact"]["get"]["responses"]["200"]["content"]
assert list(art) == ["application/octet-stream"], art
assert "412" in spec["paths"]["/v1/deployments"]["post"]["responses"] and "401" in spec["paths"]["/v1/robots"]["get"]["responses"]
rec = c.recommend(robot={"type": "arm", "compute": "jetson-orin-nx-16"}, task={"kind": "language"}, constraints={"deploy": "cloud"})
assert any(x["access"] == "external-api" for x in rec["recommendations"])
print("ok router", ids, "|", rec["recommendations"][0]["model"])

print("\nrobot        teacher(params, sim)   student(params, sim, bytes)   on-robot runtime   secs")
for t in rows:
    print(f"{t[0]:12s} {t[1]:>7d} {t[2]:>5.0%}          {t[3]:>5d} {t[4]:>5.0%} {t[5]:>6d}            {t[6]:>5.0%}          {t[7]}")
print("\nALL PASSED")
