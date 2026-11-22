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
assert p["latest_eval"]["mode"] == "offline" and p["latest_eval"]["action_mse"] < 0.01, p["latest_eval"]
dep = c.deploy(p["id"], r["id"], max_action_mse=0.01)
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
assert rec["suggested_pipeline"]
rec = c.recommend(robot={"type": "arm", "compute": "jetson-orin-nx-16"}, task={"kind": "language"}, constraints={"deploy": "cloud"})
assert any(x["access"] == "external-api" for x in rec["recommendations"])
print("ok router", ids, "|", rec["recommendations"][0]["model"])

print("\nrobot        teacher(params, sim)   student(params, sim, bytes)   on-robot runtime   secs")
for t in rows:
    print(f"{t[0]:12s} {t[1]:>7d} {t[2]:>5.0%}          {t[3]:>5d} {t[4]:>5.0%} {t[5]:>6d}            {t[6]:>5.0%}          {t[7]}")
print("\nALL PASSED")
