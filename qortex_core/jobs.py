"""Job execution. Pure functions: bytes and JSON in, bytes and JSON out.

The API server runs these on an inline thread. A remote worker (for
example a GPU box) can run the same functions after claiming a job.
"""
import io
import numpy as np
from . import envs, learn
from .policy import Policy

BACKENDS = ["native"]


def pack(obs, actions):
    buf = io.BytesIO()
    np.savez_compressed(buf, obs=np.asarray(obs, np.float32), actions=np.asarray(actions, np.float32))
    return buf.getvalue()


def unpack(blob):
    z = np.load(io.BytesIO(blob), allow_pickle=False)
    return z["obs"], z["actions"]


def run(kind, p, blobs, log=print):
    """Returns (result_dict, artifact_bytes_or_None)."""
    if kind == "generate":
        obs, act, sr = envs.collect(p["robot_type"], p["episodes"], p.get("seed", 0), p.get("noise", 0.1))
        if "dataset" in blobs:
            o0, a0 = unpack(blobs["dataset"])
            obs, act = np.concatenate([o0, obs]), np.concatenate([a0, act])
        return {"steps": int(len(obs)), "added_episodes": p["episodes"], "expert_success_rate": sr}, pack(obs, act)

    if kind == "finetune":
        obs, act = unpack(blobs["dataset"])
        pol = learn.behavior_clone(obs, act, hidden=p.get("hidden", [256, 256]), epochs=p.get("epochs", 40),
                                   seed=p.get("seed", 0), meta=p.get("meta"), log=log)
        return {"n_params": pol.n_params, "train_mse": pol.meta["train_mse"], "steps": int(len(obs))}, pol.to_bytes()

    if kind == "distill":
        teacher = Policy.from_bytes(blobs["teacher"])
        obs, _ = unpack(blobs["dataset"])
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
            obs, act = unpack(blobs["dataset"])
            n = max(1, len(obs) // 10)
            err = float(np.mean((pol.act(obs[-n:]) - np.clip(act[-n:], -1, 1)) ** 2))
            card = {"mode": "offline", "action_mse": round(err, 6), "held_out_steps": int(n), "success_rate": None,
                    "note": "No sim task for this robot type. This compares predicted actions with the last 10% of the dataset; "
                            "that slice was also used for training, so treat it as a sanity check, not validation."}
        card["n_params"] = pol.n_params
        card["quantize"] = pol.meta.get("quantize", "none")
        return card, None

    raise ValueError(f"unknown job kind {kind}")
