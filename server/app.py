"""Qortex control plane. One process: API, inline worker, docs site.

Run:  uvicorn server.app:app --port 8000
Env:  QORTEX_DATA (dir, default ./data)   QORTEX_INLINE_WORKER (1/0)
      QORTEX_WORKER_TOKEN (enables remote workers)   QORTEX_SIGNUP (1/0)
"""
import base64
import json
import os
import secrets
import sqlite3
import threading
import time
import traceback
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from qortex_core import envs, jobs
from qortex_core.policy import Policy, sha256
from . import router

DATA = os.environ.get("QORTEX_DATA", "./data")
os.makedirs(DATA, exist_ok=True)
WORKER_TOKEN = os.environ.get("QORTEX_WORKER_TOKEN", "")
ROBOT_TYPES = list(envs.ENVS) + ["custom"]

_db = sqlite3.connect(os.path.join(DATA, "qortex.db"), check_same_thread=False)
_db.execute("CREATE TABLE IF NOT EXISTS obj (id TEXT PRIMARY KEY, kind TEXT, owner TEXT, created REAL, data TEXT)")
_db.execute("CREATE INDEX IF NOT EXISTS obj_kind ON obj (kind, owner, created)")
_lock = threading.RLock()


def new_id(prefix):
    return f"{prefix}_{secrets.token_hex(6)}"


def put(kind, owner, data, id=None):
    id = id or new_id(kind[:3])
    data = dict(data, id=id, created=data.get("created", round(time.time(), 3)))
    with _lock:
        _db.execute("INSERT OR REPLACE INTO obj VALUES (?,?,?,?,?)", (id, kind, owner, data["created"], json.dumps(data)))
        _db.commit()
    return data


def get(kind, id, owner=None):
    with _lock:
        row = _db.execute("SELECT owner, data FROM obj WHERE id=? AND kind=?", (id, kind)).fetchone()
    if not row or (owner is not None and row[0] != owner):
        raise HTTPException(404, f"{kind} '{id}' not found")
    return json.loads(row[1])


def update(kind, id, **fields):
    with _lock:
        row = _db.execute("SELECT owner, data FROM obj WHERE id=? AND kind=?", (id, kind)).fetchone()
        data = dict(json.loads(row[1]), **fields)
        _db.execute("UPDATE obj SET data=? WHERE id=?", (json.dumps(data), id))
        _db.commit()
    return data


def listing(kind, owner, **where):
    with _lock:
        rows = _db.execute("SELECT data FROM obj WHERE kind=? AND owner=? ORDER BY created DESC", (kind, owner)).fetchall()
    out = [json.loads(r[0]) for r in rows]
    return [o for o in out if all(o.get(k) == v for k, v in where.items())]


def blob_path(id):
    return os.path.join(DATA, f"{id}.bin")


def write_blob(id, b):
    with open(blob_path(id), "wb") as f:
        f.write(b)


def read_blob(id):
    with open(blob_path(id), "rb") as f:
        return f.read()


# ---------------------------------------------------------------- auth

def auth(authorization: str = Header(default="")):
    key = authorization.removeprefix("Bearer ").strip()
    with _lock:
        row = _db.execute("SELECT id FROM obj WHERE kind='key' AND json_extract(data,'$.secret')=?", (key,)).fetchone()
    if not key or not row:
        raise HTTPException(401, "missing or invalid API key. Send 'Authorization: Bearer qx_...'")
    return row[0]


# ---------------------------------------------------------------- jobs

def enqueue(owner, kind, params, blobs, target=None, backend="native"):
    """blobs: {name: blob_id}. target: (obj_kind, obj_id) updated on finish."""
    return put("job", owner, {"kind": kind, "params": params, "blobs": blobs, "target": target,
                              "backend": backend, "status": "queued", "logs": [], "result": None, "error": None})


def _finish(job, result, artifact):
    """Apply a finished job's result to the object it was working on."""
    owner_row = _db.execute("SELECT owner FROM obj WHERE id=?", (job["id"],)).fetchone()
    owner = owner_row[0]
    kind, target = job["kind"], job["target"]
    if kind == "generate":
        write_blob(target[1], artifact)
        d = get("dataset", target[1])
        update("dataset", target[1], steps=result["steps"], episodes=d["episodes"] + result["added_episodes"],
               status="ready", expert_success_rate=result["expert_success_rate"])
    elif kind in ("finetune", "distill"):
        write_blob(target[1], artifact)
        pol = update("policy", target[1], status="ready", n_params=result["n_params"], sha256=sha256(artifact),
                     size_bytes=len(artifact), training=result)
        if pol.get("auto_eval", True):
            _start_eval(owner, pol, 50)
    elif kind == "eval":
        ev = update("eval", target[1], status="done", scorecard=result)
        update("policy", ev["policy_id"], latest_eval={"id": ev["id"], **result})
    update("job", job["id"], status="done", result=result, finished=round(time.time(), 3))


def _run_inline(job):
    logs = []

    def log(msg):
        logs.append(msg)
        update("job", job["id"], logs=logs[-50:])

    try:
        update("job", job["id"], status="running", started=round(time.time(), 3))
        blobs = {k: read_blob(v) for k, v in job["blobs"].items()}
        result, artifact = jobs.run(job["kind"], job["params"], blobs, log)
        _finish(job, result, artifact)
    except Exception as e:  # noqa
        _fail(job, f"{type(e).__name__}: {e}", traceback.format_exc())


def _fail(job, error, detail=""):
    update("job", job["id"], status="failed", error=error, detail=detail[-2000:])
    if job["target"]:
        update(job["target"][0], job["target"][1], status="failed", error=error)


def _claim(backends):
    with _lock:
        rows = _db.execute("SELECT data FROM obj WHERE kind='job' AND json_extract(data,'$.status')='queued' ORDER BY created").fetchall()
        for r in rows:
            j = json.loads(r[0])
            if j["backend"] in backends:
                update("job", j["id"], status="claimed")
                return j
    return None


def _worker_loop():
    while True:
        job = _claim(jobs.BACKENDS)
        if job:
            _run_inline(job)
        else:
            time.sleep(0.2)


def _start_eval(owner, pol, episodes, seed=10_000):
    mode = "sim" if pol["robot_type"] in envs.ENVS else "offline"
    ev = put("eval", owner, {"policy_id": pol["id"], "robot_type": pol["robot_type"], "mode": mode,
                            "episodes": episodes, "status": "queued", "scorecard": None})
    blobs = {"policy": pol["id"]}
    if mode == "offline":
        blobs["dataset"] = pol["dataset_id"]
    job = enqueue(owner, "eval", {"mode": mode, "robot_type": pol["robot_type"], "episodes": episodes, "seed": seed},
                  blobs, ("eval", ev["id"]))
    return update("eval", ev["id"], job_id=job["id"])


# ---------------------------------------------------------------- app

app = FastAPI(title="Qortex API", version="0.1.0", docs_url="/reference", redoc_url=None,
              description="Model routing and a cloud-to-robot policy pipeline: fine-tune, distill, evaluate in sim, deploy.")


@app.on_event("startup")
def _startup():
    if os.environ.get("QORTEX_INLINE_WORKER", "1") == "1":
        threading.Thread(target=_worker_loop, daemon=True).start()


@app.exception_handler(HTTPException)
def _http_err(_, exc):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"ok": True, "version": app.version}


class KeyIn(BaseModel):
    email: str


@app.post("/v1/keys", tags=["Auth"], summary="Create an API key")
def create_key(body: KeyIn):
    if os.environ.get("QORTEX_SIGNUP", "1") != "1":
        raise HTTPException(403, "signup is disabled on this server")
    k = put("key", "-", {"email": body.email, "secret": "qx_" + secrets.token_urlsafe(24)})
    return {"api_key": k["secret"], "note": "Store this now. It is not shown again."}


# -- catalog and router

@app.get("/v1/models", tags=["Router"], summary="List the model catalog")
def models():
    return {"models": list(router.MODELS.values()), "note": router.NOTE}


@app.get("/v1/devices", tags=["Router"], summary="List known compute targets")
def devices():
    return {"devices": list(router.DEVICES.values())}


class Task(BaseModel):
    kind: str = Field("single_task", description="single_task | multi_task | language | teacher")
    description: str = ""
    needs_vision: bool = False


class Constraints(BaseModel):
    deploy: str = Field("on_robot", description="on_robot | on_prem | cloud")
    min_hz: float = 10
    max_power_w: Optional[float] = None
    commercial: bool = True
    allow_paid: bool = True


class RecommendIn(BaseModel):
    robot_id: Optional[str] = None
    robot: Optional[dict] = Field(None, description="Inline robot spec if you have not registered one: {type, compute, power_budget_w}")
    task: Task = Task()
    constraints: Constraints = Constraints()


@app.post("/v1/recommend", tags=["Router"], summary="Rank models for a robot, task and constraints")
def recommend(body: RecommendIn, owner: str = Depends(auth)):
    robot = get("robot", body.robot_id, owner) if body.robot_id else body.robot
    if not robot or "type" not in robot:
        raise HTTPException(422, "give robot_id or robot: {type, compute}")
    if body.task.kind not in router.TASK_KINDS:
        raise HTTPException(422, f"task.kind must be one of {router.TASK_KINDS}")
    if body.constraints.deploy not in router.DEPLOY:
        raise HTTPException(422, f"constraints.deploy must be one of {router.DEPLOY}")
    measured = {}
    for p in listing("policy", owner, robot_type=robot["type"]):
        if p.get("latest_eval") and p["base_model"] not in measured:
            measured[p["base_model"]] = p["latest_eval"]
    return router.recommend(robot, body.task.model_dump(), body.constraints.model_dump(), measured)


# -- robots

class RobotIn(BaseModel):
    name: str
    type: str = Field(..., description="arm-2dof | mobile-base | drone | custom")
    compute: Optional[str] = Field(None, description="A device id from GET /v1/devices")
    power_budget_w: Optional[float] = None
    obs_dim: Optional[int] = Field(None, description="Required for type=custom")
    action_dim: Optional[int] = Field(None, description="Required for type=custom")
    notes: str = ""


@app.post("/v1/robots", tags=["Robots"], summary="Register a robot")
def create_robot(body: RobotIn, owner: str = Depends(auth)):
    if body.type not in ROBOT_TYPES:
        raise HTTPException(422, f"type must be one of {ROBOT_TYPES}")
    if body.compute and body.compute not in router.DEVICES:
        raise HTTPException(422, f"unknown compute '{body.compute}'. See GET /v1/devices")
    d = body.model_dump()
    if body.type in envs.ENVS:
        E = envs.ENVS[body.type]
        d.update(obs_dim=E.obs_dim, action_dim=E.action_dim, sim_task=E.description,
                 obs_spec=E.obs_spec, action_spec=E.action_spec, control_hz=20)
    elif not (body.obs_dim and body.action_dim):
        raise HTTPException(422, "custom robots need obs_dim and action_dim")
    else:
        d.update(sim_task=None)
    return put("robot", owner, d)


@app.get("/v1/robots", tags=["Robots"], summary="List robots")
def list_robots(owner: str = Depends(auth)):
    return {"robots": listing("robot", owner)}


@app.get("/v1/robots/{id}", tags=["Robots"], summary="Get a robot")
def get_robot(id: str, owner: str = Depends(auth)):
    return get("robot", id, owner)


# -- datasets

class DatasetIn(BaseModel):
    robot_id: str
    name: str = "dataset"


class Episode(BaseModel):
    observations: list[list[float]]
    actions: list[list[float]]


class EpisodesIn(BaseModel):
    episodes: list[Episode]


class GenerateIn(BaseModel):
    episodes: int = Field(150, ge=1, le=2000)
    noise: float = Field(0.1, ge=0, le=1, description="Std of noise added to executed actions, to teach recovery")
    seed: int = 0


@app.post("/v1/datasets", tags=["Datasets"], summary="Create an empty dataset")
def create_dataset(body: DatasetIn, owner: str = Depends(auth)):
    r = get("robot", body.robot_id, owner)
    return put("dataset", owner, {"robot_id": r["id"], "robot_type": r["type"], "name": body.name, "episodes": 0,
                                  "steps": 0, "status": "empty", "obs_dim": r["obs_dim"], "action_dim": r["action_dim"]})


@app.post("/v1/datasets/{id}/episodes", tags=["Datasets"], summary="Upload recorded episodes")
def add_episodes(id: str, body: EpisodesIn, owner: str = Depends(auth)):
    import numpy as np
    d = get("dataset", id, owner)
    try:
        obs = np.concatenate([np.asarray(e.observations, np.float32) for e in body.episodes])
        act = np.concatenate([np.asarray(e.actions, np.float32) for e in body.episodes])
    except ValueError as e:
        raise HTTPException(422, f"ragged episode data: {e}")
    if obs.ndim != 2 or obs.shape[1] != d["obs_dim"] or act.ndim != 2 or act.shape[1] != d["action_dim"] or len(obs) != len(act):
        raise HTTPException(422, f"expected observations [T,{d['obs_dim']}] and actions [T,{d['action_dim']}] of equal length, "
                                 f"got {obs.shape} and {act.shape}")
    if np.abs(act).max() > 1.0001:
        raise HTTPException(422, "actions must be normalised to [-1, 1]")
    if d["steps"]:
        o0, a0 = jobs.unpack(read_blob(id))
        obs, act = np.concatenate([o0, obs]), np.concatenate([a0, act])
    write_blob(id, jobs.pack(obs, act))
    return update("dataset", id, steps=int(len(obs)), episodes=d["episodes"] + len(body.episodes), status="ready")


@app.post("/v1/datasets/{id}/generate", tags=["Datasets"], summary="Generate demonstrations in sim")
def generate(id: str, body: GenerateIn, owner: str = Depends(auth)):
    d = get("dataset", id, owner)
    if d["robot_type"] not in envs.ENVS:
        raise HTTPException(422, "sim generation is only available for built-in robot types. Upload episodes instead.")
    blobs = {"dataset": id} if d["steps"] else {}
    job = enqueue(owner, "generate", dict(body.model_dump(), robot_type=d["robot_type"]), blobs, ("dataset", id))
    return update("dataset", id, status="generating", job_id=job["id"])


@app.get("/v1/datasets/{id}", tags=["Datasets"], summary="Get a dataset")
def get_dataset(id: str, owner: str = Depends(auth)):
    return get("dataset", id, owner)


# -- policies

class PolicyIn(BaseModel):
    robot_id: str
    dataset_id: str
    base_model: str = "qortex/mlp-bc"
    name: str = "policy"
    hidden: list[int] = [256, 256]
    epochs: int = Field(40, ge=1, le=500)
    auto_eval: bool = True


class DistillIn(BaseModel):
    hidden: list[int] = Field([32, 32], description="Student hidden layer sizes")
    quantize: str = Field("int8", description="int8 | none")
    epochs: int = Field(60, ge=1, le=500)
    auto_eval: bool = True


@app.post("/v1/policies", tags=["Policies"], summary="Fine-tune a policy from a base model")
def create_policy(body: PolicyIn, owner: str = Depends(auth)):
    r, d = get("robot", body.robot_id, owner), get("dataset", body.dataset_id, owner)
    if body.base_model not in router.MODELS:
        raise HTTPException(422, f"unknown base_model. See GET /v1/models")
    m = router.MODELS[body.base_model]
    if m["access"] == "external-api":
        raise HTTPException(422, f"{m['id']} is a vendor-hosted API and cannot be fine-tuned here")
    if d["status"] != "ready" or not d["steps"]:
        raise HTTPException(409, f"dataset is '{d['status']}'. It must be ready and non-empty")
    if d["robot_id"] != r["id"]:
        raise HTTPException(422, "dataset belongs to a different robot")
    backend = "native" if m["runnable"] else "gpu:" + m["id"]
    pol = put("policy", owner, {"name": body.name, "robot_id": r["id"], "robot_type": r["type"], "dataset_id": d["id"],
                                "base_model": m["id"], "parent_policy_id": None, "stage": "finetuned", "quantize": "none",
                                "status": "queued" if m["runnable"] else "waiting_for_gpu_worker",
                                "auto_eval": body.auto_eval, "latest_eval": None})
    job = enqueue(owner, "finetune", {"hidden": body.hidden, "epochs": body.epochs,
                                      "meta": {"policy_id": pol["id"], "base_model": m["id"], "robot_type": r["type"]}},
                  {"dataset": d["id"]}, ("policy", pol["id"]), backend)
    pol = update("policy", pol["id"], job_id=job["id"])
    if not m["runnable"]:
        pol["message"] = (f"No worker with a backend for {m['id']} is connected. The job stays queued until one is. "
                          "Use base_model=qortex/mlp-bc to run the full pipeline today.")
    return pol


@app.post("/v1/policies/{id}/distill", tags=["Policies"], summary="Distill a smaller student policy")
def distill(id: str, body: DistillIn, owner: str = Depends(auth)):
    t = get("policy", id, owner)
    if t["status"] != "ready":
        raise HTTPException(409, f"teacher policy is '{t['status']}', not ready")
    if body.quantize not in ("int8", "none"):
        raise HTTPException(422, "quantize must be int8 or none")
    pol = put("policy", owner, {"name": t["name"] + "-student", "robot_id": t["robot_id"], "robot_type": t["robot_type"],
                                "dataset_id": t["dataset_id"], "base_model": t["base_model"], "parent_policy_id": t["id"],
                                "stage": "distilled", "quantize": body.quantize, "status": "queued",
                                "auto_eval": body.auto_eval, "latest_eval": None})
    job = enqueue(owner, "distill", {"hidden": body.hidden, "epochs": body.epochs, "quantize": body.quantize,
                                     "meta": {"policy_id": pol["id"], "parent_policy_id": t["id"],
                                              "base_model": t["base_model"], "robot_type": t["robot_type"]}},
                  {"teacher": t["id"], "dataset": t["dataset_id"]}, ("policy", pol["id"]))
    return update("policy", pol["id"], job_id=job["id"])


@app.get("/v1/policies", tags=["Policies"], summary="List policies")
def list_policies(robot_id: Optional[str] = None, owner: str = Depends(auth)):
    return {"policies": listing("policy", owner, **({"robot_id": robot_id} if robot_id else {}))}


@app.get("/v1/policies/{id}", tags=["Policies"], summary="Get a policy and its lineage")
def get_policy(id: str, owner: str = Depends(auth)):
    return get("policy", id, owner)


@app.get("/v1/policies/{id}/artifact", tags=["Policies"], summary="Download the policy artifact (.npz)")
def artifact(id: str, owner: str = Depends(auth)):
    p = get("policy", id, owner)
    if p["status"] != "ready":
        raise HTTPException(409, f"policy is '{p['status']}'")
    return Response(read_blob(id), media_type="application/octet-stream",
                    headers={"X-Qortex-Sha256": p["sha256"], "Content-Disposition": f'attachment; filename="{id}.npz"'})


class ActIn(BaseModel):
    observation: list[float]


_policy_cache: dict[str, Policy] = {}


@app.post("/v1/policies/{id}/act", tags=["Policies"], summary="Hosted inference: observation in, action out")
def act(id: str, body: ActIn, owner: str = Depends(auth)):
    p = get("policy", id, owner)
    if p["status"] != "ready":
        raise HTTPException(409, f"policy is '{p['status']}'")
    if id not in _policy_cache:
        _policy_cache[id] = Policy.from_bytes(read_blob(id))
    pol = _policy_cache[id]
    if len(body.observation) != pol.meta["obs_dim"]:
        raise HTTPException(422, f"observation must have {pol.meta['obs_dim']} values")
    t0 = time.perf_counter()
    a = pol.act(body.observation)
    return {"action": [round(float(x), 6) for x in a], "policy_id": id, "inference_ms": round((time.perf_counter() - t0) * 1000, 3)}


# -- evals

class EvalIn(BaseModel):
    policy_id: str
    episodes: int = Field(50, ge=1, le=1000)
    seed: int = 10_000


@app.post("/v1/evals", tags=["Evals"], summary="Evaluate a policy in sim")
def create_eval(body: EvalIn, owner: str = Depends(auth)):
    p = get("policy", body.policy_id, owner)
    if p["status"] != "ready":
        raise HTTPException(409, f"policy is '{p['status']}'")
    return _start_eval(owner, p, body.episodes, body.seed)


@app.get("/v1/evals/{id}", tags=["Evals"], summary="Get an eval and its scorecard")
def get_eval(id: str, owner: str = Depends(auth)):
    return get("eval", id, owner)


# -- deployments

class Gate(BaseModel):
    min_success_rate: Optional[float] = Field(0.8, description="Sim success rate the policy's latest eval must reach")
    max_action_mse: Optional[float] = Field(None, description="For custom robots evaluated offline")


class DeployIn(BaseModel):
    policy_id: str
    robot_id: str
    target: str = Field("on_robot", description="on_robot | cloud")
    gate: Gate = Gate()
    force: bool = Field(False, description="Deploy even if the gate fails. Recorded on the deployment.")


@app.post("/v1/deployments", tags=["Deployments"], summary="Deploy a policy to a robot, gated on its eval")
def deploy(body: DeployIn, owner: str = Depends(auth)):
    p, r = get("policy", body.policy_id, owner), get("robot", body.robot_id, owner)
    if p["status"] != "ready":
        raise HTTPException(409, f"policy is '{p['status']}'")
    if p["robot_type"] != r["type"] or p["robot_id"] != r["id"]:
        raise HTTPException(422, "policy was trained for a different robot")
    if body.target not in ("on_robot", "cloud"):
        raise HTTPException(422, "target must be on_robot or cloud")
    ev, passed, reason = p.get("latest_eval"), True, "gate passed"
    if not ev:
        passed, reason = False, "policy has no eval yet"
    elif ev.get("mode") == "sim" and body.gate.min_success_rate is not None and ev["success_rate"] < body.gate.min_success_rate:
        passed, reason = False, f"sim success {ev['success_rate']:.0%} is below the gate {body.gate.min_success_rate:.0%}"
    elif ev.get("mode") == "offline" and body.gate.max_action_mse is not None and ev["action_mse"] > body.gate.max_action_mse:
        passed, reason = False, f"offline action MSE {ev['action_mse']} is above the gate {body.gate.max_action_mse}"
    elif ev.get("mode") == "offline" and body.gate.max_action_mse is None:
        reason = "no sim task for this robot type and no max_action_mse set: deployed without a real gate"
    if not passed and not body.force:
        raise HTTPException(412, f"deployment blocked: {reason}. Pass force=true to override.")
    prev = [d for d in listing("deployment", owner, robot_id=r["id"]) if d["active"]]
    for d in prev:
        update("deployment", d["id"], active=False)
    dep = put("deployment", owner, {"policy_id": p["id"], "robot_id": r["id"], "target": body.target, "active": True,
                                   "version": len(listing("deployment", owner, robot_id=r["id"])) + 1,
                                   "gate": body.gate.model_dump(), "gate_passed": passed, "gate_reason": reason,
                                   "forced": bool(body.force and not passed), "sha256": p["sha256"],
                                   "eval": ev, "previous_deployment_id": prev[0]["id"] if prev else None})
    return dep


def _current(owner, robot_id):
    act_ = [d for d in listing("deployment", owner, robot_id=robot_id) if d["active"]]
    return act_[0] if act_ else None


@app.get("/v1/robots/{id}/deployment", tags=["Deployments"], summary="Current deployment for a robot (polled by the runtime)")
def current(id: str, owner: str = Depends(auth)):
    get("robot", id, owner)
    d = _current(owner, id)
    if not d:
        raise HTTPException(404, "no active deployment for this robot")
    return dict(d, artifact_path=f"/v1/policies/{d['policy_id']}/artifact")


@app.post("/v1/robots/{id}/rollback", tags=["Deployments"], summary="Roll back to the previous deployment")
def rollback(id: str, owner: str = Depends(auth)):
    d = _current(owner, id)
    if not d or not d.get("previous_deployment_id"):
        raise HTTPException(409, "nothing to roll back to")
    update("deployment", d["id"], active=False, rolled_back=True)
    return update("deployment", d["previous_deployment_id"], active=True)


@app.get("/v1/robots/{id}/deployments", tags=["Deployments"], summary="Deployment history for a robot")
def history(id: str, owner: str = Depends(auth)):
    return {"deployments": listing("deployment", owner, robot_id=id)}


# -- telemetry

class TelemetryIn(BaseModel):
    policy_id: Optional[str] = None
    deployment_id: Optional[str] = None
    steps: int = 0
    loop_hz: Optional[float] = None
    inference_ms_p50: Optional[float] = None
    inference_ms_p99: Optional[float] = None
    episodes: Optional[int] = None
    successes: Optional[int] = None
    extra: dict[str, Any] = {}


@app.post("/v1/robots/{id}/telemetry", tags=["Telemetry"], summary="Report runtime stats from the robot")
def telemetry(id: str, body: TelemetryIn, owner: str = Depends(auth)):
    get("robot", id, owner)
    return put("telemetry", owner, dict(body.model_dump(), robot_id=id), id=new_id("tel"))


@app.get("/v1/robots/{id}/telemetry", tags=["Telemetry"], summary="Recent telemetry for a robot")
def get_telemetry(id: str, limit: int = 20, owner: str = Depends(auth)):
    return {"telemetry": listing("telemetry", owner, robot_id=id)[:limit]}


# -- jobs

@app.get("/v1/jobs/{id}", tags=["Jobs"], summary="Get a job. Use ?wait=30 to long-poll until it finishes")
def get_job(id: str, wait: float = 0, owner: str = Depends(auth)):
    end = time.time() + min(wait, 60)
    while True:
        j = get("job", id, owner)
        if j["status"] in ("done", "failed") or time.time() >= end:
            j.pop("blobs", None)
            return j
        time.sleep(0.25)


# -- remote workers (optional)

def _worker_auth(x_worker_token: str = Header(default="")):
    if not WORKER_TOKEN or not secrets.compare_digest(x_worker_token, WORKER_TOKEN):
        raise HTTPException(401, "invalid worker token")


class ClaimIn(BaseModel):
    backends: list[str] = ["native"]


@app.post("/v1/worker/claim", include_in_schema=False, dependencies=[Depends(_worker_auth)])
def worker_claim(body: ClaimIn):
    job = _claim(body.backends)
    if not job:
        return {"job": None}
    return {"job": {"id": job["id"], "kind": job["kind"], "params": job["params"], "backend": job["backend"],
                    "blobs": {k: base64.b64encode(read_blob(v)).decode() for k, v in job["blobs"].items()}}}


class FinishIn(BaseModel):
    result: Optional[dict] = None
    artifact_b64: Optional[str] = None
    error: Optional[str] = None


@app.post("/v1/worker/jobs/{id}/finish", include_in_schema=False, dependencies=[Depends(_worker_auth)])
def worker_finish(id: str, body: FinishIn):
    job = get("job", id)
    if body.error:
        _fail(job, body.error)
    else:
        _finish(job, body.result, base64.b64decode(body.artifact_b64) if body.artifact_b64 else None)
    return {"ok": True}


# -- docs site (built by mkdocs into ./site)
_site = os.path.join(os.path.dirname(os.path.dirname(__file__)), "site")
if os.path.isdir(_site):
    app.mount("/", StaticFiles(directory=_site, html=True), name="docs")
