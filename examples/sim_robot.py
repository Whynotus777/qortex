"""A stand-in robot: runs the on-robot runtime against the local MuJoCo task.
Use it to test a deployment before you have hardware.

    python examples/sim_robot.py --robot-id rob_xxx --type drone --episodes 10
Needs: pip install mujoco numpy, plus QORTEX_URL and QORTEX_API_KEY.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdk"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qortex import Agent, Client          # noqa: E402
from qortex_core import envs              # noqa: E402


def run(client, robot_id, robot_type, episodes=10, seed=50_000, cache_dir="~/.qortex"):
    agent = Agent(client, robot_id, cache_dir=cache_dir)
    if not agent.sync():
        raise SystemExit("no active deployment for this robot")
    env, wins = envs.make(robot_type), 0
    for ep in range(episodes):
        obs, done = env.reset(seed + ep), False
        while not done:
            obs, done, info = env.step(agent.act(obs))
        wins += info["success"]
    stats = agent.report(episodes=episodes, successes=wins, extra={"source": "examples/sim_robot.py"})
    return wins / episodes, stats, agent


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-id", required=True)
    ap.add_argument("--type", required=True, choices=list(envs.ENVS))
    ap.add_argument("--episodes", type=int, default=10)
    a = ap.parse_args()
    sr, stats, agent = run(Client(), a.robot_id, a.type, a.episodes)
    print(f"policy {agent.deployment['policy_id']}: {sr:.0%} success over {a.episodes} episodes")
    print(stats)
