# Mobile base (`mobile-base`)

A holonomic base driven by planar velocity commands.

| | |
|---|---|
| Observation (6) | `[x, y, vx, vy, goal_x, goal_y]` in metres and m/s, in a fixed world frame |
| Action (2) | `[vx, vy]`. Command `velocity = 1.0 m/s * action` |
| Control rate | 20 Hz |
| Sim task | Drive to a random goal up to about 3 m away; success is within 8 cm |

## Real hardware

```python
from qortex import Agent, Client

def read_obs():
    x, y, vx, vy = read_odometry()        # world frame
    gx, gy = current_goal()
    return [x, y, vx, vy, gx, gy]

def send_action(a):
    command_velocity(vx=1.0 * a[0], vy=1.0 * a[1])     # world-frame m/s

Agent(Client(), "rob_...").run(read_obs, send_action, hz=20)
```

The commands are in the world frame. Rotate them into the body frame before sending them to a base that expects body velocities. A differential-drive base cannot follow `vy` directly; convert the velocity vector to a heading and forward speed, or use a [custom robot](custom.md).

The sim has no obstacles. Keep your own collision avoidance and e-stop in the loop.
