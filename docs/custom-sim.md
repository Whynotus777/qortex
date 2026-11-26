# Bring your own sim

Upload your own MuJoCo model and a short task description, and a custom robot gets the same pipeline as the built-in types: sim data generation, closed-loop evals and a success-rate deploy gate.

No code is uploaded or run. The task is a JSON spec that names joints, actuators, sites and bodies already in your model.

```python
from qortex import Client
qx = Client()

robot = qx.create_robot("my-arm", type="custom", compute="jetson-orin-nano", sim={
    "mjcf": open("my_arm.xml").read(),
    "task": {
        "control_hz": 20,
        "horizon": 100,
        "point": {"site": "tip"},
        "goal": {"via_qpos": {"low": [-2.3, 0.4], "high": [2.3, 2.4]}},
        "reset": {"qpos_low": [-1.0, 0.5], "qpos_high": [1.0, 2.0]},
        "success": {"tolerance": 0.03},
        "action": {"mode": "delta_qpos", "scale": 0.2},
        "observation": [{"type": "qpos"}, {"type": "qvel", "scale": 0.1}, {"type": "point"}, {"type": "goal"}],
    },
})
print(robot["obs_dim"], robot["action_dim"], robot["sim_info"]["sanity"])

ds      = qx.generate(qx.create_dataset(robot["id"])["id"], episodes=120)
teacher = qx.finetune(robot["id"], ds["id"])
print(teacher["latest_eval"]["success_rate"])       # closed loop, in your sim
student = qx.distill(teacher["id"])
qx.deploy(student["id"], robot["id"], min_success_rate=0.9)
```

`obs_dim` and `action_dim` are derived from the sim, so you do not pass them.

## The task spec

The task is always "move a point to a goal".

| Field | Meaning |
|---|---|
| `point` | The point that must reach the goal: `{"site": name}` or `{"body": name}` |
| `goal` | How goals are sampled. `{"via_qpos": {"low": [...], "high": [...]}}` puts the model in a random joint configuration and uses where the point ends up, so every goal is reachable. `{"box": {"low": [x,y,z], "high": [x,y,z]}}` samples a position in a box |
| `success.tolerance` | Distance in metres between point and goal at the end of the episode. Default 0.03 |
| `reset` | Optional `{"qpos_low": [...], "qpos_high": [...]}` for the start pose. Default is the model's initial pose |
| `control_hz` | Control rate. Default 20. Physics steps per control step are derived from the model's timestep |
| `horizon` | Control steps per episode. Default 150, maximum 1000 |
| `action.mode` | `delta_qpos`: each actuator's command is its joint position plus `scale` times the action. Needs position actuators on hinge or slide joints. `ctrl`: the action in [-1, 1] is mapped onto each actuator's `ctrlrange` |
| `observation` | A list of terms, concatenated in order |

Observation terms, each with an optional `"scale"`:

| Type | Values |
|---|---|
| `qpos`, `qvel` | All joint positions or velocities |
| `sensors` | All sensor readings defined in the model |
| `site_pos`, `body_pos` | World position of the named site or body (needs `"name"`) |
| `point` | World position of the task point |
| `goal` | The goal position |
| `goal_delta` | Goal minus point |

Vectors under `goal.via_qpos` and `reset` have one value per position coordinate in the model (`nq`).

## What you get back

The robot's `sim_info` reports the derived sizes, the observation layout, and a sanity check run at upload:

```json
{"obs_dim": 10, "action_dim": 2, "obs_spec": "qpos[2] + qvel[2] + point[3] + goal[3]",
 "has_expert": true, "sanity": {"expert_success_rate": 1.0, "random_success_rate": 0.0}}
```

If `expert_success_rate` is low, the task is too hard as specified: raise the horizon or tolerance, or check actuator gains. If `random_success_rate` is high, the task is too easy to mean anything.

## Generating demonstrations

`qx.generate(...)` works when the sim has a built-in expert, which needs `action.mode` `delta_qpos` and `goal.via_qpos`. The expert is a joint-space controller that drives toward the goal configuration. For any other task, upload your own episodes; observations must follow `obs_spec`.

## Attach, replace, download, delete

```python
qx.attach_sim(robot_id, mjcf, task)     # the new sim must produce the same obs_dim and action_dim
qx.get_sim(robot_id)                    # {"mjcf": ..., "task": ..., "info": ...}
qx.delete_robot(robot_id)               # removes the robot, its sim, datasets, policies and deployments
```

Run a deployed policy against your sim locally, through the real runtime:

```bash
python examples/sim_robot.py --robot-id rob_... --type custom --episodes 10
```

## Limits

- One self-contained MJCF file up to 1 MB. No `<include>`, no mesh, texture or heightfield files, no plugins. Use primitive geoms for now.
- At most 64 position coordinates and 32 actuators.
- One task shape: reach a goal with a point. No contact-rich, multi-stage or vision tasks.
- A policy that passes in your sim has been tested against your model of the robot, not the robot. The sim is only as good as its masses, gains and limits.
