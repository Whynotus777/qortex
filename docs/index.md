# Qortex

Qortex takes a robot policy from data to a running robot through one API:

1. **Route.** Describe the robot, the task and your constraints. Get a ranked list of models, with the evidence behind each rank.
2. **Fine-tune.** Train a policy on your dataset, or on demonstrations generated in sim.
3. **Distill.** Shrink the policy to fit the robot's onboard computer.
4. **Evaluate.** Every policy is rolled out in sim and gets a scorecard.
5. **Deploy.** Ship to the robot only if the scorecard passes your gate. Roll back with one call.

```python
from qortex import Client

qx = Client()                                    # QORTEX_URL and QORTEX_API_KEY
robot   = qx.create_robot("scout", type="drone", compute="jetson-orin-nx-16")
pick    = qx.recommend(robot_id=robot["id"])["recommendations"][0]
data    = qx.generate(qx.create_dataset(robot["id"])["id"], episodes=150)
teacher = qx.finetune(robot["id"], data["id"], base_model=pick["model"])
student = qx.distill(teacher["id"], hidden=(32, 32), quantize="int8")
qx.deploy(student["id"], robot["id"], min_success_rate=0.9)
```

[Start the quickstart](quickstart.md){ .md-button .md-button--primary }
[API reference](api-reference.md){ .md-button }

## What works today

| Area | Status |
|---|---|
| Router over open and paid models | Working. Third-party models are ranked from published specs, not from our own benchmarks. |
| Fine-tune, distill, quantize | Working for the built-in `qortex/mlp-bc` policy family. |
| Sim evals | Working for `arm-2dof`, `mobile-base` and `drone`. |
| Bring your own sim | Working: upload a self-contained MuJoCo model and a task spec for a custom robot. Reach-a-goal tasks, primitive geoms only. |
| Gated deploys, rollback, on-robot runtime, telemetry | Working. |
| Hosted inference | Working for policies trained here. |
| Training VLA models (SmolVLA, pi0, GR00T, OpenVLA) | Not yet. Jobs are accepted and wait for a GPU worker backend that does not exist yet. |

Read [Status and limits](limits.md) before you put a policy on real hardware.
