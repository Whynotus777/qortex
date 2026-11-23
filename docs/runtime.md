# On-robot runtime

The runtime is a small Python agent. Its only dependency is numpy, so it runs on a Jetson, a Raspberry Pi or a laptop.

```bash
pip install ./sdk          # from the repo
export QORTEX_URL=https://qortex.quantumroboticslab.ai QORTEX_API_KEY=qx_...
```

```python
from qortex import Agent, Client

def read_obs():
    return [...]                 # floats, in the order given by the robot's obs_spec

def send_action(a):
    ...                          # a is in [-1, 1]; scale it for your actuators

Agent(Client(), "rob_...").run(read_obs, send_action, hz=20)
```

## What it does

- **Pulls the active deployment** and verifies the artifact's SHA-256 before loading it.
- **Hot-swaps** when you deploy a new policy or roll back. It checks every 5 seconds.
- **Keeps running offline.** Artifacts are cached in `~/.qortex`. If the server is unreachable the loop continues with the current policy, and on restart it loads the last cached one.
- **Reports telemetry** every 10 seconds: steps, loop rate, inference latency p50 and p99.

## Manual control of the loop

```python
agent = Agent(Client(), "rob_...")
agent.sync()
while True:
    action = agent.act(read_obs())
    send_action(action)
```

Call `agent.sync()` yourself when you want to check for a new deployment, and `agent.report()` to send telemetry.

## Hosted inference instead

For slow loops or quick tests you can skip the runtime and call the cloud:

```python
action = qx.act(policy_id, observation)
```

Every call is a network round trip, so do not use this for real-time control.

## Loading an artifact with no SDK

```python
from qortex.policy import Policy
policy = Policy.load("pol_xxx.npz")
action = policy.act(obs)
```

`sdk/qortex/policy.py` is a single file you can copy onto the robot.
