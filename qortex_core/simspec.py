"""Bring-your-own sim: a MuJoCo model plus a declarative task spec.

No user code is executed. The spec only names things that already exist in
the model (joints, actuators, sites, bodies) and says how to read
observations, apply actions, sample goals and score success.

Task spec (JSON):

  control_hz   int, default 20
  horizon      int steps per episode, default 150
  point        {"site": name} or {"body": name}: the point that must reach the goal
  goal         how goals are sampled, one of
                 {"via_qpos": {"low": [...], "high": [...]}}   put the model in a random joint
                                                              configuration and use where the
                                                              point ends up (always reachable)
                 {"box": {"low": [x,y,z], "high": [x,y,z]}}    a random position in a box
  success      {"tolerance": metres}: the point is within this distance of the goal at the end
  reset        optional {"qpos_low": [...], "qpos_high": [...]}; default is the model's initial pose
  action       {"mode": "delta_qpos", "scale": 0.2}   ctrl = joint position + scale * action
               {"mode": "ctrl"}                       ctrl spans each actuator's ctrlrange
  observation  list of terms, concatenated in order:
                 {"type": "qpos"} {"type": "qvel"} {"type": "sensors"}
                 {"type": "site_pos", "name": ...} {"type": "body_pos", "name": ...}
                 {"type": "point"} {"type": "goal"} {"type": "goal_delta"}
               each may carry "scale": number
"""
import re

import numpy as np
import mujoco

MAX_XML_BYTES = 1_000_000
MAX_NQ, MAX_NU, MAX_HORIZON, MAX_SUBSTEPS = 64, 32, 1000, 20_000
OBS_TYPES = {"qpos", "qvel", "sensors", "site_pos", "body_pos", "point", "goal", "goal_delta"}


class SpecError(ValueError):
    pass


def check_xml(xml):
    """Reject models that reach outside themselves. Only self-contained MJCF is accepted."""
    if not isinstance(xml, str) or not xml.strip():
        raise SpecError("mjcf must be a non-empty string of MJCF XML")
    if len(xml.encode()) > MAX_XML_BYTES:
        raise SpecError(f"mjcf is larger than {MAX_XML_BYTES} bytes")
    low = xml.lower()
    for bad, why in (("<!doctype", "DOCTYPE declarations"), ("<!entity", "XML entities"), ("<include", "<include> elements"),
                     ("<plugin", "plugins"), ("<extension", "extensions")):
        if bad in low:
            raise SpecError(f"{why} are not allowed: the model must be a single self-contained file")
    if re.search(r"\b(file|meshdir|texturedir|assetdir)\s*=", low):
        raise SpecError("external files (meshes, textures, heightfields) are not supported yet: use primitive geoms")


def _vec(v, n, what):
    a = np.asarray(v, np.float64)
    if a.shape != (n,):
        raise SpecError(f"{what} must have {n} values, got {a.size}")
    return a


class SpecEnv:
    def __init__(self, mjcf, task):
        check_xml(mjcf)
        if not isinstance(task, dict):
            raise SpecError("task must be an object")
        try:
            self.model = mujoco.MjModel.from_xml_string(mjcf)
        except Exception as e:  # noqa
            raise SpecError(f"MuJoCo could not load the model: {str(e)[:300]}")
        m = self.model
        self.data = mujoco.MjData(m)
        if m.nq > MAX_NQ or m.nu > MAX_NU:
            raise SpecError(f"model too large: at most {MAX_NQ} position coordinates and {MAX_NU} actuators")
        if m.nu == 0:
            raise SpecError("the model has no actuators")
        self.task = task
        hz = float(task.get("control_hz", 20))
        if not 1 <= hz <= 500:
            raise SpecError("control_hz must be between 1 and 500")
        self.control_hz = hz
        self.frame_skip = max(1, int(round(1.0 / (hz * m.opt.timestep))))
        self.horizon = int(task.get("horizon", 150))
        if not 1 <= self.horizon <= MAX_HORIZON:
            raise SpecError(f"horizon must be between 1 and {MAX_HORIZON}")
        if self.horizon * self.frame_skip > MAX_SUBSTEPS:
            raise SpecError(f"episode too long: horizon x physics steps per control step must be at most {MAX_SUBSTEPS}")

        self._point = self._locator(task.get("point"), "point")

        goal = task.get("goal") or {}
        if "via_qpos" in goal:
            g = goal["via_qpos"]
            self._gq = (_vec(g.get("low"), m.nq, "goal.via_qpos.low"), _vec(g.get("high"), m.nq, "goal.via_qpos.high"))
            self._gbox = None
        elif "box" in goal:
            g = goal["box"]
            self._gbox = (_vec(g.get("low"), 3, "goal.box.low"), _vec(g.get("high"), 3, "goal.box.high"))
            self._gq = None
        else:
            raise SpecError("goal must be {'via_qpos': {low, high}} or {'box': {low, high}}")

        self.tol = float((task.get("success") or {}).get("tolerance", 0.03))
        if self.tol <= 0:
            raise SpecError("success.tolerance must be positive")

        rs = task.get("reset") or {}
        self._reset_q = (_vec(rs["qpos_low"], m.nq, "reset.qpos_low"), _vec(rs["qpos_high"], m.nq, "reset.qpos_high")) \
            if "qpos_low" in rs else None

        act = task.get("action") or {"mode": "ctrl"}
        self.mode = act.get("mode", "ctrl")
        if self.mode == "delta_qpos":
            trn = m.actuator_trntype
            if not np.all(trn == mujoco.mjtTrn.mjTRN_JOINT):
                raise SpecError("action.mode delta_qpos needs every actuator to drive a joint")
            jid = m.actuator_trnid[:, 0]
            if not np.all(np.isin(m.jnt_type[jid], [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE])):
                raise SpecError("action.mode delta_qpos supports hinge and slide joints only")
            self._qadr = m.jnt_qposadr[jid]
            self._scale = np.broadcast_to(np.asarray(act.get("scale", 0.2), np.float64), (m.nu,)).copy()
        elif self.mode == "ctrl":
            if not np.all(m.actuator_ctrllimited):
                raise SpecError("action.mode ctrl needs a ctrlrange on every actuator, so [-1, 1] can be mapped onto it")
        else:
            raise SpecError("action.mode must be 'delta_qpos' or 'ctrl'")
        self.action_dim = int(m.nu)

        self.rng = np.random.default_rng(0)
        self.goal = np.zeros(3)
        self.goal_q = None
        self.t = 0
        mujoco.mj_forward(m, self.data)
        terms = task.get("observation")
        if not isinstance(terms, list) or not terms:
            raise SpecError("observation must be a non-empty list of terms")
        self._terms, layout = [], []
        for t in terms:
            ty = (t or {}).get("type")
            if ty not in OBS_TYPES:
                raise SpecError(f"unknown observation type '{ty}'. Allowed: {sorted(OBS_TYPES)}")
            fn = self._term(t)
            self._terms.append((fn, float(t.get("scale", 1.0))))
            layout.append(f"{ty}{'(' + str(t['name']) + ')' if 'name' in t else ''}[{np.size(fn())}]")
        self.obs_dim = int(self.reset(0).size)
        self.obs_spec = " + ".join(layout)
        self.action_spec = (f"{m.nu} joint position deltas, each scaled by {self._scale.tolist()}" if self.mode == "delta_qpos"
                            else f"{m.nu} actuator commands, [-1, 1] mapped onto each ctrlrange")
        self.description = task.get("description") or "uploaded sim: move the point to a sampled goal"
        self.has_expert = self.mode == "delta_qpos" and self._gq is not None

    # -- helpers
    def _locator(self, spec, what):
        m = self.model
        if not isinstance(spec, dict) or not (("site" in spec) ^ ("body" in spec)):
            raise SpecError(f"{what} must be {{'site': name}} or {{'body': name}}")
        kind, name = ("site", spec["site"]) if "site" in spec else ("body", spec["body"])
        obj = mujoco.mjtObj.mjOBJ_SITE if kind == "site" else mujoco.mjtObj.mjOBJ_BODY
        i = mujoco.mj_name2id(m, obj, str(name))
        if i < 0:
            raise SpecError(f"{what}: no {kind} named '{name}' in the model")
        return (lambda: self.data.site_xpos[i]) if kind == "site" else (lambda: self.data.xpos[i])

    def _term(self, t):
        ty, d = t["type"], self.data
        if ty == "qpos":
            return lambda: d.qpos
        if ty == "qvel":
            return lambda: d.qvel
        if ty == "sensors":
            if self.model.nsensordata == 0:
                raise SpecError("observation type 'sensors' used but the model defines no sensors")
            return lambda: d.sensordata
        if ty in ("site_pos", "body_pos"):
            return self._locator({ty.split("_")[0]: t.get("name")}, f"observation {ty}")
        if ty == "point":
            return self._point
        if ty == "goal":
            return lambda: self.goal
        return lambda: self.goal - self._point()

    def _clip_q(self, q):
        m = self.model
        for j in range(m.njnt):
            if m.jnt_limited[j] and m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                a = m.jnt_qposadr[j]
                q[a] = np.clip(q[a], *m.jnt_range[j])
        return q

    # -- env interface (same as qortex_core.envs)
    def reset(self, seed=None):
        m, d = self.model, self.data
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(m, d)
        self.t = 0
        if self._gq is not None:
            self.goal_q = self._clip_q(self.rng.uniform(*self._gq))
            d.qpos[:] = self.goal_q
            mujoco.mj_forward(m, d)
            self.goal = np.array(self._point(), np.float64)
        else:
            self.goal_q = None
            self.goal = self.rng.uniform(*self._gbox)
        mujoco.mj_resetData(m, d)
        if self._reset_q is not None:
            d.qpos[:] = self._clip_q(self.rng.uniform(*self._reset_q))
        mujoco.mj_forward(m, d)
        return self.obs()

    def obs(self):
        return np.concatenate([np.atleast_1d(np.asarray(fn(), np.float64)) * s for fn, s in self._terms]).astype(np.float32)

    def dist(self):
        return float(np.linalg.norm(np.asarray(self._point()) - self.goal))

    def step(self, action):
        m, d = self.model, self.data
        a = np.clip(np.asarray(action, np.float64).reshape(-1), -1, 1)
        if a.size != m.nu:
            raise SpecError(f"action must have {m.nu} values")
        if self.mode == "delta_qpos":
            ctrl = d.qpos[self._qadr] + self._scale * a
            lim = m.actuator_ctrllimited.astype(bool)
            ctrl[lim] = np.clip(ctrl[lim], m.actuator_ctrlrange[lim, 0], m.actuator_ctrlrange[lim, 1])
        else:
            lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
            ctrl = lo + (a + 1) / 2 * (hi - lo)
        d.ctrl[:] = ctrl
        for _ in range(self.frame_skip):
            mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)):
            raise SpecError("the simulation went unstable (non-finite state). Check actuator gains and the timestep")
        self.t += 1
        dd = self.dist()
        return self.obs(), self.t >= self.horizon, {"dist": dd, "success": bool(dd < self.tol)}

    def expert(self):
        """Joint-space P controller toward the goal configuration. Only for delta_qpos + via_qpos tasks."""
        if not self.has_expert:
            raise SpecError("no built-in expert for this task: it needs action.mode delta_qpos and goal.via_qpos")
        return np.clip((self.goal_q[self._qadr] - self.data.qpos[self._qadr]) / self._scale, -1, 1)


def describe(env):
    return {"obs_dim": env.obs_dim, "action_dim": env.action_dim, "obs_spec": env.obs_spec, "action_spec": env.action_spec,
            "control_hz": env.control_hz, "horizon": env.horizon, "physics_steps_per_control_step": env.frame_skip,
            "success_tolerance_m": env.tol, "has_expert": env.has_expert, "nq": int(env.model.nq), "nu": int(env.model.nu)}


def sanity(env, episodes=5):
    """Quick solvability check run at upload time."""
    out = {}
    rng = np.random.default_rng(7)
    for name in (["expert"] if env.has_expert else []) + ["random"]:
        wins = 0
        for ep in range(episodes):
            env.reset(900_000 + ep)
            done = False
            while not done:
                _, done, info = env.step(env.expert() if name == "expert" else rng.uniform(-1, 1, env.action_dim))
            wins += info["success"]
        out[f"{name}_success_rate"] = wins / episodes
    return out
