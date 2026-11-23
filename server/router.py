"""Model router: rank catalog models for a robot, a task and constraints.

Scores for third-party models come from published specs, not measurement.
Every result says which kind of evidence it rests on.
"""
import json
import os

_CAT = json.load(open(os.path.join(os.path.dirname(__file__), "catalog.json")))
MODELS = {m["id"]: m for m in _CAT["models"]}
DEVICES = {d["id"]: d for d in _CAT["devices"]}
NOTE = _CAT["_note"]

TASK_KINDS = ["single_task", "multi_task", "language", "teacher"]
DEPLOY = ["on_robot", "on_prem", "cloud"]


def _est_hz(model, device):
    if not model.get("params_b"):
        return None
    return round(device["speed"] / model["params_b"], 1)


def recommend(robot, task, constraints, measured=None):
    """measured: {model_id: latest sim scorecard} from this account, if any."""
    measured = measured or {}
    kind = task.get("kind", "single_task")
    deploy = constraints.get("deploy", "on_robot")
    min_hz = constraints.get("min_hz", 10)
    device = DEVICES.get(robot.get("compute") or "", None)
    power = constraints.get("max_power_w") or robot.get("power_budget_w")
    needs_vision = task.get("needs_vision", False)
    ranked, excluded = [], []

    for m in MODELS.values():
        why, caveats = [], list(m.get("caveats", []))
        if robot["type"] not in m["embodiments"] and "any" not in m["embodiments"]:
            excluded.append({"model": m["id"], "reason": f"no support listed for robot type '{robot['type']}'"})
            continue
        if constraints.get("commercial", True) and not m["commercial_ok"]:
            excluded.append({"model": m["id"], "reason": "licence does not allow commercial use"})
            continue
        if m["access"] == "external-api":
            if not constraints.get("allow_paid", True):
                excluded.append({"model": m["id"], "reason": "paid APIs excluded by constraint"})
                continue
            if deploy != "cloud":
                excluded.append({"model": m["id"], "reason": f"vendor-hosted only, cannot deploy {deploy}"})
                continue
        if kind == "language" and not m["language"]:
            excluded.append({"model": m["id"], "reason": "no language conditioning"})
            continue
        if needs_vision and "vision" not in m["inputs"]:
            excluded.append({"model": m["id"], "reason": "no camera input"})
            continue

        score = m["task_fit"].get(kind, 0.3)
        why.append(f"task fit for '{kind}': {m['task_fit'].get(kind, 0.3):.2f}")
        fit, hz = "cloud", None

        if deploy == "on_robot" and kind != "teacher":
            if device is None:
                caveats.append("robot has no known compute set, so on-robot fit was not checked")
                fit = "unchecked"
            else:
                hz = _est_hz(m, device)
                mem = m["infer_mem_gb"]
                if mem <= device["mem_gb"] * 0.7 and (device["gpu"] or mem < 0.5):
                    fit = "direct"
                    why.append(f"fits {device['name']} as is (about {mem} GB of {device['mem_gb']} GB)")
                elif mem * 0.4 <= device["mem_gb"] * 0.7 and device["gpu"]:
                    fit = "needs_quantization"
                    score -= 0.1
                    why.append(f"fits {device['name']} only after int8/4-bit quantization")
                else:
                    fit = "needs_distillation"
                    score -= 0.25
                    why.append(f"too large for {device['name']}: use as a cloud teacher and distill a student")
                if hz is not None and fit != "needs_distillation":
                    if hz >= min_hz:
                        why.append(f"estimated {hz} Hz meets the {min_hz} Hz target")
                    else:
                        score -= 0.2 * min(1.0, (min_hz - hz) / min_hz)
                        caveats.append(f"estimated {hz} Hz is below the {min_hz} Hz target; action chunking or distillation needed")
                if power and device.get("watts") and device["watts"] > power:
                    caveats.append(f"{device['name']} draws up to {device['watts']} W, over the {power} W budget")
                if power and power <= 15 and m["params_b"] and m["params_b"] < 0.01:
                    score += 0.1
                    why.append("tiny model: negligible compute draw, good for battery life")
        elif deploy == "on_prem":
            fit = "on_prem_gpu" if m["infer_mem_gb"] and m["infer_mem_gb"] > 0.5 else "direct"
        elif kind == "teacher":
            fit = "cloud_teacher"

        ev = "published-specs"
        e = measured.get(m["id"])
        if e and e.get("mode") == "sim" and e.get("success_rate") is not None:
            ev = "measured"
            score += 0.2 * e["success_rate"]
            why.append(f"measured {e['success_rate']:.0%} success in your closed-loop sim evals")
        elif e and e.get("mode") == "offline":
            ev = "offline-check"
            score += 0.05 if m["runnable"] else 0
            why.append(f"offline action MSE {e.get('action_mse')} on "
                       f"{'held-out episodes' if e.get('validated') else 'training data'} for this robot type; not a closed-loop result")
        elif m["runnable"]:
            score += 0.05
            why.append("can be trained and evaluated on this platform today")
        if not m["runnable"] and m["access"] == "open-weights":
            caveats.append("training this model needs a GPU worker with its backend, not yet available here")

        ranked.append({
            "model": m["id"], "name": m["name"], "access": m["access"], "license": m["license"],
            "score": round(max(score, 0), 3), "fit": fit, "evidence": ev,
            "estimated_hz": hz, "hz_is_estimate": hz is not None,
            "runnable_here": m["runnable"], "why": why, "caveats": caveats,
        })

    ranked.sort(key=lambda r: -r["score"])
    pipeline = None
    if ranked and deploy == "on_robot":
        top = ranked[0]
        teachers = [r for r in ranked if r["fit"] == "needs_distillation"]
        if top["fit"] in ("direct", "unchecked"):
            pipeline = [f"fine-tune {top['model']} on your dataset", "evaluate in sim", f"deploy to robot ({top['fit']})"]
        elif top["fit"] == "needs_quantization":
            pipeline = [f"fine-tune {top['model']}", "quantize to int8", "evaluate the quantized build in sim", "deploy to robot"]
        if teachers and teachers[0]["score"] + 0.25 > top["score"]:
            students = [r["model"] for r in ranked if r["fit"] == "direct"][:1]
            pipeline = [f"fine-tune {teachers[0]['model']} in the cloud as teacher",
                        f"distill into {students[0] if students else 'a smaller student'}",
                        "evaluate student against teacher in sim", "deploy student to robot"]
    blocked = [mid for mid in MODELS if pipeline and not MODELS[mid]["runnable"] and any(mid in st for st in pipeline)]
    runnable = [r for r in ranked if r["runnable_here"]]
    alternative = None
    if blocked and runnable:
        alternative = {"model": runnable[0]["model"],
                       "steps": [f"fine-tune {runnable[0]['model']} on your dataset", "evaluate", "deploy to robot"],
                       "tradeoff": "; ".join(runnable[0]["caveats"]) or None}
    return {"recommendations": ranked, "excluded": excluded, "suggested_pipeline": pipeline,
            "pipeline_executable_here": bool(pipeline) and not blocked,
            "pipeline_blocked_by": [{"model": mid, "reason": "no training backend for this model on this platform yet"} for mid in blocked],
            "executable_alternative": alternative,
            "executable_alternative_note": None if not blocked or alternative else
                "No model that can be trained on this platform today meets these requirements.",
            "inputs": {"robot_type": robot["type"], "compute": robot.get("compute"), "task_kind": kind,
                       "deploy": deploy, "min_hz": min_hz, "power_budget_w": power},
            "note": NOTE}
