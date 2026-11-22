"""On-robot runtime. Pulls the active deployment, runs the control loop,
hot-swaps when a new version is deployed, and reports telemetry.

    agent = Agent(client, robot_id)
    agent.run(read_obs, send_action, hz=20)

read_obs() -> list/array of floats matching the robot's obs spec
send_action(a) receives an array in [-1, 1]; scale it to your actuators.
"""
import hashlib
import os
import time

import numpy as np

from .client import QortexError
from .policy import Policy


class Agent:
    def __init__(self, client, robot_id, cache_dir="~/.qortex", poll_s=5.0, telemetry_s=10.0):
        self.client, self.robot_id = client, robot_id
        self.cache_dir = os.path.expanduser(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.poll_s, self.telemetry_s = poll_s, telemetry_s
        self.policy, self.deployment = None, None
        self._lat, self._steps, self._t_poll, self._t_tel, self._t_start = [], 0, 0.0, time.time(), time.time()

    def sync(self):
        """Fetch the active deployment if it changed. Returns True on a swap.
        If the server is unreachable, keeps running the cached policy."""
        self._t_poll = time.time()
        try:
            dep = self.client.current_deployment(self.robot_id)
        except QortexError as e:
            if e.status == 404:
                return False
            raise
        except OSError:
            return self._load_cached()
        if self.deployment and dep["id"] == self.deployment["id"]:
            return False
        path = os.path.join(self.cache_dir, f"{dep['sha256']}.npz")
        if not os.path.exists(path):
            blob = self.client.download_artifact(dep["policy_id"])
            if hashlib.sha256(blob).hexdigest() != dep["sha256"]:
                raise RuntimeError("artifact checksum mismatch, refusing to load")
            with open(path + ".tmp", "wb") as f:
                f.write(blob)
            os.replace(path + ".tmp", path)
        self.policy, self.deployment = Policy.load(path), dep
        with open(os.path.join(self.cache_dir, f"{self.robot_id}.current"), "w") as f:
            f.write(path)
        return True

    def _load_cached(self):
        if self.policy is not None:
            return False
        ptr = os.path.join(self.cache_dir, f"{self.robot_id}.current")
        if os.path.exists(ptr):
            self.policy = Policy.load(open(ptr).read().strip())
            self.deployment = {"id": "cached", "policy_id": self.policy.meta.get("policy_id")}
            return True
        return False

    def act(self, obs):
        if self.policy is None:
            raise RuntimeError("no policy loaded. Deploy one to this robot, then call sync()")
        t0 = time.perf_counter()
        a = self.policy.act(obs)
        self._lat.append((time.perf_counter() - t0) * 1000)
        self._steps += 1
        return a

    def report(self, **extra):
        if not self._lat:
            return
        lat = np.array(self._lat)
        stats = dict(policy_id=self.deployment.get("policy_id"), deployment_id=self.deployment.get("id"),
                     steps=self._steps, loop_hz=round(self._steps / max(time.time() - self._t_start, 1e-6), 2),
                     inference_ms_p50=round(float(np.percentile(lat, 50)), 4),
                     inference_ms_p99=round(float(np.percentile(lat, 99)), 4), **extra)
        self._lat, self._steps, self._t_tel, self._t_start = [], 0, time.time(), time.time()
        try:
            self.client.telemetry(self.robot_id, **stats)
        except (OSError, QortexError):
            pass                       # never let reporting stop the control loop
        return stats

    def run(self, read_obs, send_action, hz=20, max_steps=None):
        dt, n = 1.0 / hz, 0
        self.sync()
        while max_steps is None or n < max_steps:
            t0 = time.time()
            if t0 - self._t_poll > self.poll_s:
                try:
                    if self.sync():
                        print(f"[qortex] loaded policy {self.deployment['policy_id']}")
                except Exception as e:  # noqa
                    print(f"[qortex] sync failed, keeping current policy: {e}")
            if self.policy is not None:
                send_action(self.act(read_obs()))
                n += 1
            if t0 - self._t_tel > self.telemetry_s:
                self.report()
            time.sleep(max(0.0, dt - (time.time() - t0)))
        self.report()
