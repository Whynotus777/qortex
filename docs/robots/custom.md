# Custom robot

!!! tip
    If you have a MuJoCo model of your robot, [upload it](../custom-sim.md). You then get sim data generation and closed-loop evals instead of the offline check described below.

Use `custom` for any robot that is not one of the built-in types. You choose the observation and action sizes and upload your own data.

```python
robot = qx.create_robot("my-arm", type="custom", obs_dim=14, action_dim=7, compute="jetson-orin-nano")
ds = qx.create_dataset(robot["id"])

qx.upload_episodes(ds["id"], [
    {"observations": [[...14 floats...], ...],     # one row per timestep
     "actions":      [[...7 floats...], ...]},     # same length, each value in [-1, 1]
])

policy = qx.finetune(robot["id"], ds["id"])
print(policy["latest_eval"])          # mode: offline
qx.deploy(policy["id"], robot["id"], max_action_mse=0.01)
```

## Rules for the data

- Observations are flat float vectors of length `obs_dim`. Put joint angles, velocities, goal coordinates and any other state in a fixed order.
- Actions must be normalised to `[-1, 1]`. Divide by your actuator limits before uploading and multiply back in `send_action`.
- Upload in several calls if you like; episodes are appended.

## How custom policies are validated

There is no sim task for a custom robot, so Qortex holds data back instead:

- Before training, the last 20% of your **episodes** (at least one) are reserved. Whole episodes, never individual steps.
- The policy and any distilled student never train on them.
- The offline eval scores action error on those reserved episodes only. The scorecard reports `validated: true`, `held_out_episodes` and `scored_steps`.
- A dataset with a single episode cannot be validated. The scorecard says `validated: false`, and the deploy gate refuses it unless you pass `force=True`.

Upload each real run as its own episode so the split is meaningful.

## What is different from built-in types

| | Built-in types | Custom |
|---|---|---|
| Sim data generation | Yes | No, upload recordings |
| Eval | Closed-loop sim rollouts | Offline: action error on held-out episodes |
| Deploy gate | `min_success_rate` | `max_action_mse` (required) |

!!! warning
    Matching recorded actions on unseen episodes is a real check, but it is not proof the robot completes the task. Small errors compound in closed loop. Test on hardware with a hand on the e-stop.

The policy family available today takes state vectors only. It cannot use camera images.
