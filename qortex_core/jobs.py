"""Job execution. Pure functions: bytes and JSON in, bytes and JSON out.

The API server runs these on an inline thread. A remote worker (for
example a GPU box) can run the same functions after claiming a job.
"""
import io
import numpy as np
from . import envs, learn
from .policy import Policy

BACKENDS = ["native"]


def pack(obs, actions, ep_lens=None):
    arrs = {"obs": np.asarray(obs, np.float32), "actions": np.asarray(actions, np.float32)}
    arrs["ep_lens"] = np.asarray(ep_lens if ep_lens is not None else [len(arrs["obs"])], np.int64)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrs)
    return buf.getvalue()


def unpack(blob):
    z = np.load(io.BytesIO(blob), allow_pickle=False)
    return z["obs"], z["actions"]


def ep_lens(blob):
    z = np.load(io.BytesIO(blob), allow_pickle=False)
    return z["ep_lens"] if "ep_lens" in z else np.array([len(z["obs"])], np.int64)


def split(blob, holdout_episodes):
    """Reserve the last N whole episodes. Returns (train_obs, train_act, held_obs, held_act, n_held_episodes)."""
    obs, act = unpack(blob)
    lens = ep_lens(blob)
    h = int(max(0, min(holdout_episodes or 0, len(lens) - 1)))
    cut = len(obs) - int(lens[-h:].sum()) if h else len(obs)
    return obs[:cut], act[:cut], obs[cut:], act[cut:], h


def run(kind, p, blobs, log=print):
    """Returns (result_dict, artifact_bytes_or_None)."""
    if kind == "generate":
        obs, act, sr = envs.collect(p["robot_type"], p["episodes"], p.get("seed", 0), p.get("noise", 0.1))
        lens = np.full(p["episodes"], envs.ENVS[p["robot_type"]].horizon, np.int64)
        if "dataset" in blobs:
            o0, a0 = unpack(blobs["dataset"])
            obs, act = np.concatenate([o0, obs]), np.concatenate([a0, act])
            lens = np.concatenate([ep_lens(blobs["dataset"]), lens])
        return {"steps": int(len(obs)), "added_episodes": p["episodes"], "expert_success_rate": sr}, pack(obs, act, lens)

    if kind == "finetune":
        obs, act, _, _, held = split(blobs["dataset"], p.get("holdout_episodes", 0))
        pol = learn.behavior_clone(obs, act, hidden=p.get("hidden", [256, 256]), epochs=p.get("epochs", 40),
                                   seed=p.get("seed", 0), meta=p.get("meta"), log=log)
        return {"n_params": pol.n_params, "train_mse": pol.meta["train_mse"], "train_steps": int(len(obs)),
                "held_out_episodes": held}, pol.to_bytes()

    if kind == "distill":
        teacher = Policy.from_bytes(blobs["teacher"])
        obs = split(blobs["dataset"], p.get("holdout_episodes", 0))[0]
        pol = learn.distill(teacher, obs, hidden=p.get("hidden", [32, 32]), epochs=p.get("epochs", 60),
                            seed=p.get("seed", 0), meta=p.get("meta"), log=log)
        blob = pol.to_bytes(p.get("quantize", "int8"))
        return {"n_params": pol.n_params, "distill_mse": pol.meta["distill_mse"],
                "teacher_params": teacher.n_params,
                "compression": round(teacher.n_params / pol.n_params, 1)}, blob

    if kind == "eval":
        pol = Policy.from_bytes(blobs["policy"])
        if p["mode"] == "sim":
            card = envs.evaluate(p["robot_type"], pol, p.get("episodes", 50), p.get("seed", 10_000))
            card["mode"] = "sim"
            card["task"] = envs.ENVS[p["robot_type"]].description
        else:
            _, _, obs, act, held = split(blobs["dataset"], p.get("holdout_episodes", 0))
            validated = held > 0
            if not validated:          # nothing was reserved: score on training data and say so
                obs, act = unpack(blobs["dataset"])
            err = float(np.mean((pol.act(obs) - np.clip(act, -1, 1)) ** 2))
            card = {"mode": "offline", "action_mse": round(err, 6), "success_rate": None, "validated": validated,
                    "held_out_episodes": held, "scored_steps": int(len(obs)),
                    "note": ("Action error on whole episodes that were reserved before training. No sim task exists for this "
                             "robot type, so this is not a closed-loop result.") if validated else
                            ("NOT validated: the dataset has a single episode, so nothing could be held out and this score "
                             "is on training data. Upload at least two episodes.")}
        card["n_params"] = pol.n_params
        card["quantize"] = pol.meta.get("quantize", "none")
        return card, None

    raise ValueError(f"unknown job kind {kind}")
