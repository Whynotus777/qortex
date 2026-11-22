# Arm (`arm-2dof`)

A planar arm with two revolute joints, each 0.2 m long, driven by joint position targets.

| | |
|---|---|
| Observation (10) | `[cos q1, sin q1, cos q2, sin q2, dq1/10, dq2/10, target_x, target_y, tip_x, tip_y]` |
| Action (2) | Joint position deltas. Command `q_target = q + 0.2 * action` radians |
| Control rate | 20 Hz |
| Joint limits | q1 in [-2.6, 2.6], q2 in [0.2, 2.6] rad |
| Sim task | Reach a random target; success is tip within 3 cm |

## Real hardware

```python
import math
from qortex import Agent, Client

def read_obs():
    q1, q2, dq1, dq2 = read_joint_state()          # your driver
    tx, ty = current_target()                      # metres, in the arm base frame
    x = 0.2 * (math.cos(q1) + math.cos(q1 + q2))
    y = 0.2 * (math.sin(q1) + math.sin(q1 + q2))
    return [math.cos(q1), math.sin(q1), math.cos(q2), math.sin(q2), dq1 / 10, dq2 / 10, tx, ty, x, y]

def send_action(a):
    q1, q2, *_ = read_joint_state()
    command_joint_positions(q1 + 0.2 * a[0], q2 + 0.2 * a[1])

Agent(Client(), "rob_...").run(read_obs, send_action, hz=20)
```

The policy assumes 0.2 m links. If your links differ, scale target and tip positions by `0.2 / your_link_length`, or register a [custom robot](custom.md) and train on your own data.

For arms with more joints, cameras or grippers, use a [custom robot](custom.md).
