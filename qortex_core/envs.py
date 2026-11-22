"""MuJoCo sim tasks, one per built-in robot type.

Each env exposes: reset(seed) -> obs, step(action) -> (obs, done, info),
expert() -> action, plus obs_dim / action_dim. Actions are always in [-1, 1].

These are deliberately small. They exist so the whole pipeline
(data -> train -> distill -> eval -> deploy) can run on a CPU.
"""
import numpy as np
import mujoco

ARM_XML = """
<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.01" gravity="0 0 0"/>
  <worldbody>
    <body name="l1" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 0 1" damping="0.4" range="-2.6 2.6" limited="true"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.3"/>
      <body name="l2" pos="0.2 0 0">
        <joint name="j2" type="hinge" axis="0 0 1" damping="0.4" range="0.2 2.6" limited="true"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.3"/>
        <site name="tip" pos="0.2 0 0" size="0.01"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position joint="j1" kp="30"/>
    <position joint="j2" kp="30"/>
  </actuator>
</mujoco>
"""

BASE_XML = """
<mujoco>
  <option timestep="0.01" gravity="0 0 0"/>
  <worldbody>
    <body name="base" pos="0 0 0.05">
      <joint name="x" type="slide" axis="1 0 0" damping="1"/>
      <joint name="y" type="slide" axis="0 1 0" damping="1"/>
      <geom type="cylinder" size="0.15 0.05" mass="5"/>
    </body>
  </worldbody>
  <actuator>
    <velocity joint="x" kv="40"/>
    <velocity joint="y" kv="40"/>
  </actuator>
</mujoco>
"""

DRONE_XML = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <body name="drone" pos="0 0 1" gravcomp="1">
      <joint name="x" type="slide" axis="1 0 0" damping="0.3"/>
      <joint name="y" type="slide" axis="0 1 0" damping="0.3"/>
      <joint name="z" type="slide" axis="0 0 1" damping="0.3"/>
      <geom type="box" size="0.2 0.2 0.04" mass="1.5"/>
    </body>
  </worldbody>
  <actuator>
    <velocity joint="x" kv="12"/>
    <velocity joint="y" kv="12"/>
    <velocity joint="z" kv="12"/>
  </actuator>
</mujoco>
"""


class _Env:
    xml = ""
    frame_skip = 5
    horizon = 100
    obs_dim = 0
    action_dim = 0
    tol = 0.03
    description = ""
    obs_spec = ""
    action_spec = ""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(0)
        self.t = 0

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.t = 0
        self._reset()
        mujoco.mj_forward(self.model, self.data)
        return self.obs()

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        self._apply(a)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.t += 1
        d = self.dist()
        done = self.t >= self.horizon
        return self.obs(), done, {"dist": d, "success": bool(d < self.tol)}


class ArmReach(_Env):
    """2-DOF planar arm, position-controlled joints, reach a target point."""
    xml = ARM_XML
    obs_dim, action_dim, tol, horizon = 10, 2, 0.03, 100
    description = "2-DOF planar arm reaching a target with its tip"
    obs_spec = "[cos q1, sin q1, cos q2, sin q2, dq1/10, dq2/10, target_x, target_y, tip_x, tip_y]"
    action_spec = "[dq1, dq2] joint position deltas, scaled by 0.2 rad"

    def _fk(self, q):
        return 0.2 * np.array([np.cos(q[0]) + np.cos(q[0] + q[1]), np.sin(q[0]) + np.sin(q[0] + q[1])])

    def _reset(self):
        self.data.qpos[:] = [self.rng.uniform(-1, 1), self.rng.uniform(0.5, 2.0)]
        self.q_goal = np.array([self.rng.uniform(-2.3, 2.3), self.rng.uniform(0.4, 2.4)])
        self.target = self._fk(self.q_goal)

    def tip(self):
        return self.data.site_xpos[0][:2].copy()

    def obs(self):
        q, dq = self.data.qpos, self.data.qvel
        return np.concatenate([[np.cos(q[0]), np.sin(q[0]), np.cos(q[1]), np.sin(q[1])],
                               dq / 10.0, self.target, self.tip()]).astype(np.float32)

    def _apply(self, a):
        self.data.ctrl[:] = self.data.qpos + 0.2 * a

    def dist(self):
        return float(np.linalg.norm(self.tip() - self.target))

    def expert(self):
        return np.clip((self.q_goal - self.data.qpos) / 0.2, -1, 1)


class BaseNav(_Env):
    """Holonomic mobile base driven by planar velocity commands."""
    xml = BASE_XML
    obs_dim, action_dim, tol, horizon = 6, 2, 0.08, 120
    vmax = 1.0
    description = "Holonomic mobile base driving to a goal position"
    obs_spec = "[x, y, vx, vy, goal_x, goal_y] in metres and m/s"
    action_spec = "[vx, vy] velocity command, scaled by 1.0 m/s"

    def _reset(self):
        self.data.qpos[:] = self.rng.uniform(-0.5, 0.5, 2)
        self.goal = self.rng.uniform(-2, 2, 2)

    def obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel, self.goal]).astype(np.float32)

    def _apply(self, a):
        self.data.ctrl[:] = self.vmax * a

    def dist(self):
        return float(np.linalg.norm(self.data.qpos - self.goal))

    def expert(self):
        return np.clip(2.0 * (self.goal - self.data.qpos) / self.vmax, -1, 1)


class DroneWaypoint(BaseNav):
    """Drone flown by velocity setpoints, as in PX4 offboard mode.

    This models the vehicle as a velocity-controlled body. It does not
    simulate rotors or attitude; the autopilot is assumed to do that.
    """
    xml = DRONE_XML
    obs_dim, action_dim, tol, horizon = 9, 3, 0.10, 150
    vmax = 1.5
    description = "Drone flying to a 3D waypoint using velocity setpoints (offboard-style)"
    obs_spec = "[x, y, z, vx, vy, vz, goal_x, goal_y, goal_z] relative to the reset pose, metres and m/s"
    action_spec = "[vx, vy, vz] velocity setpoint, scaled by 1.5 m/s"

    def _reset(self):
        self.data.qpos[:] = self.rng.uniform(-0.3, 0.3, 3)
        self.goal = np.concatenate([self.rng.uniform(-3, 3, 2), self.rng.uniform(-0.5, 2.0, 1)])


ENVS = {"arm-2dof": ArmReach, "mobile-base": BaseNav, "drone": DroneWaypoint}


def make(robot_type):
    if robot_type not in ENVS:
        raise KeyError(f"no sim task for robot type '{robot_type}'")
    return ENVS[robot_type]()


def collect(robot_type, episodes, seed=0, noise=0.1):
    """Roll out the scripted expert. Returns (obs, actions, success_rate).

    Noise is added to the executed action, while the clean expert action is
    recorded. That gives the learner examples of recovering from mistakes.
    """
    env = make(robot_type)
    rng = np.random.default_rng(seed + 12345)
    O, A, wins = [], [], 0
    for ep in range(episodes):
        o = env.reset(seed + ep)
        done = False
        while not done:
            a = env.expert()
            O.append(o)
            A.append(a)
            o, done, info = env.step(a + noise * rng.standard_normal(a.shape))
        wins += info["success"]
    return np.array(O, np.float32), np.array(A, np.float32), wins / max(episodes, 1)


def evaluate(robot_type, policy_fn, episodes=50, seed=10_000):
    """Closed-loop rollouts of policy_fn(obs) -> action on unseen seeds."""
    import time
    env = make(robot_type)
    wins, dists, lat = 0, [], []
    for ep in range(episodes):
        o = env.reset(seed + ep)
        done = False
        while not done:
            t0 = time.perf_counter()
            a = policy_fn(o)
            lat.append(time.perf_counter() - t0)
            o, done, info = env.step(a)
        wins += info["success"]
        dists.append(info["dist"])
    lat = np.array(lat) * 1000
    return {
        "episodes": episodes,
        "success_rate": round(wins / episodes, 4),
        "mean_final_distance_m": round(float(np.mean(dists)), 4),
        "latency_ms_p50": round(float(np.percentile(lat, 50)), 4),
        "latency_ms_p99": round(float(np.percentile(lat, 99)), 4),
        "max_control_hz_on_eval_host": int(1000 / max(np.percentile(lat, 99), 1e-6)),
        "success_tolerance_m": env.tol,
    }
