# Custom robot

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

## What is different from built-in types

| | Built-in types | Custom |
|---|---|---|
| Sim data generation | Yes | No, upload recordings |
| Eval | Closed-loop sim rollouts | Offline: action error on the last 10% of the dataset |
| Deploy gate | `min_success_rate` | `max_action_mse` |

!!! warning
    The offline eval is a sanity check. The slice it scores was also used for training, and matching recorded actions does not prove the robot completes the task. Test on hardware with a hand on the e-stop.

The policy family available today takes state vectors only. It cannot use camera images.
