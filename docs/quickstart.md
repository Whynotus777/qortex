# Quickstart

You will train a policy, shrink it, test it in sim and run it through the on-robot runtime. No hardware is needed. It takes about five minutes.

## 1. Install the SDK

```bash
git clone https://github.com/Whynotus777/qortex && cd qortex
pip install ./sdk mujoco
```

`mujoco` is only needed for the simulated robot in step 7.

## 2. Get an API key

```bash
export QORTEX_URL=https://qortex.quantumroboticslab.ai
curl -s -X POST $QORTEX_URL/v1/keys -H 'content-type: application/json' \
  -d '{"email": "you@example.com"}'
export QORTEX_API_KEY=qx_...
```

The SDK reads both variables. `QORTEX_URL` defaults to `https://qortex.quantumroboticslab.ai`, so you only need to set it for a self-hosted server.

## 3. Register a robot

```python
from qortex import Client
qx = Client()

robot = qx.create_robot("scout", type="drone", compute="jetson-orin-nx-16", power_budget_w=40)
print(robot["obs_spec"], robot["action_spec"])
```

The built-in types are `arm-2dof`, `mobile-base` and `drone`. Each one has a sim task. For anything else use [`custom`](robots/custom.md).

## 4. Ask the router for a model

```python
rec = qx.recommend(robot_id=robot["id"],
                   task={"kind": "single_task"},
                   constraints={"deploy": "on_robot", "min_hz": 20})
for r in rec["recommendations"][:3]:
    print(r["model"], r["score"], r["fit"], r["evidence"])
```

## 5. Build a dataset

Generate demonstrations in sim:

```python
ds = qx.create_dataset(robot["id"])
ds = qx.generate(ds["id"], episodes=150)
```

Or upload your own recordings with `qx.upload_episodes(...)`. See [Custom robot](robots/custom.md) for the format.

## 6. Fine-tune, distill, deploy

```python
teacher = qx.finetune(robot["id"], ds["id"], base_model="qortex/mlp-bc")
print(teacher["latest_eval"]["success_rate"])       # sim eval runs automatically

student = qx.distill(teacher["id"], hidden=(32, 32), quantize="int8")
print(student["n_params"], student["latest_eval"]["success_rate"])

qx.deploy(student["id"], robot["id"], min_success_rate=0.9)
```

If the student's sim success rate is below the gate, the deploy call fails with HTTP 412 and nothing ships.

## 7. Run it on a (simulated) robot

```bash
python examples/sim_robot.py --robot-id rob_... --type drone --episodes 10
```

This script is the same runtime you would put on a real robot. It downloads the deployed artifact, checks its SHA-256, runs the control loop against a local MuJoCo task and posts telemetry.

## 8. Put it on real hardware

```python
from qortex import Agent, Client

agent = Agent(Client(), robot_id="rob_...")
agent.run(read_obs, send_action, hz=20)
```

You write `read_obs` and `send_action` for your robot. See the page for your [robot type](robots/drone.md) and the [runtime guide](runtime.md).
