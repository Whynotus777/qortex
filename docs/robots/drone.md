# Drone (`drone`)

The policy outputs velocity setpoints, the same interface as PX4 offboard mode. Your autopilot keeps doing attitude and rate control.

| | |
|---|---|
| Observation (9) | `[x, y, z, vx, vy, vz, goal_x, goal_y, goal_z]`, metres and m/s, **z up**, relative to where the policy was started |
| Action (3) | `[vx, vy, vz]`. Command `velocity = 1.5 m/s * action` |
| Control rate | 20 Hz |
| Sim task | Fly to a random waypoint within about ±3 m horizontally and -0.5 to +2 m vertically; success is within 10 cm |

!!! danger "Before flying"
    The sim models the vehicle as a velocity-controlled body. It has no rotors, attitude dynamics, wind, battery sag or obstacles. A policy that passes the sim gate has **not** been validated for flight.

    - Test in PX4 SITL first.
    - First real flights: props off, then tethered or netted, with a pilot on the RC kill switch.
    - Clamp the commanded velocity and geofence in your own code. Do not rely on the policy for limits.
    - Goals outside the training range (above) are outside what the policy has seen.

## PX4 with MAVSDK

This adapter has not been flown on hardware. Treat it as a starting point.

```python
import asyncio
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw
from qortex import Agent, Client

VMAX = 1.5
GOAL = [2.0, 1.0, 1.0]            # metres from the start pose, z up

async def main():
    drone = System()
    await drone.connect(system_address="udp://:14540")
    agent = Agent(Client(), "rob_...")
    agent.sync()

    origin = None
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
    await drone.offboard.start()

    async for pv in drone.telemetry.position_velocity_ned():
        p, v = pv.position, pv.velocity
        if origin is None:
            origin = (p.north_m, p.east_m, p.down_m)
        # NED -> x north, y east, z up, relative to start
        obs = [p.north_m - origin[0], p.east_m - origin[1], -(p.down_m - origin[2]),
               v.north_m_s, v.east_m_s, -v.down_m_s, *GOAL]
        a = agent.act(obs)
        vx, vy, vz = (max(-1, min(1, float(x))) * VMAX for x in a)
        await drone.offboard.set_velocity_ned(VelocityNedYaw(vx, vy, -vz, 0.0))

asyncio.run(main())
```

Run the telemetry stream at 20 Hz to match training (`set_rate_position_velocity_ned(20)`).

## Test without a vehicle

```bash
python examples/sim_robot.py --robot-id rob_... --type drone --episodes 10
```
