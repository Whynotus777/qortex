"""GPU worker: trains LeRobot policies (ACT today) for Qortex jobs.

Two ways to run it. Both need the same three settings:

    QORTEX_URL            your Qortex host (default https://qortex.quantumroboticslab.ai)
    QORTEX_WORKER_TOKEN   the worker token set on the server
    HF_TOKEN              a Hugging Face token with write access (weights are pushed to a private model repo)

1. Your own GPU box (for example an RTX 5090), with lerobot installed:

       python -m worker.gpu                 # loop forever
       python -m worker.gpu --once          # run one job and exit

2. Modal (serverless GPUs; nothing runs, and nothing is billed, while no job is queued):

       modal secret create qortex-worker QORTEX_WORKER_TOKEN=... HF_TOKEN=... QORTEX_URL=...
       modal deploy worker/gpu.py

   A small CPU function checks for jobs every two minutes and starts a GPU container per job.

How a job runs: claim it with a lease, run `lerobot-train`, send a heartbeat every 30 s (progress + lease renewal),
upload the final checkpoint folder to the Hub, report `hf://user/repo` back. Checkpoints go to a persistent
directory keyed by job id, so a preempted or crashed run resumes from the last checkpoint when the lease expires
and the job is claimed again.

This file is self-contained on purpose (stdlib only, plus lerobot/huggingface_hub at run time) so Modal can ship
it as a single module.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

BACKENDS = ["gpu:lerobot"]
LEASE_S = 600
BEAT_S = 30
RUNS_DIR = Path(os.environ.get("QORTEX_RUNS_DIR", "runs"))


def _cfg():
    return (os.environ.get("QORTEX_URL", "https://qortex.quantumroboticslab.ai").rstrip("/"),
            os.environ["QORTEX_WORKER_TOKEN"])


def call(path, body, timeout=60):
    url, token = _cfg()
    req = urllib.request.Request(url + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "X-Worker-Token": token})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def claim(worker):
    return call("/v1/worker/claim", {"backends": BACKENDS, "worker": worker, "lease_seconds": LEASE_S})["job"]


# -- lerobot-train command and log parsing

def build_command(params, out_dir, device="cuda", extra=()):
    """A fresh run, or a resume when an earlier attempt left a checkpoint behind."""
    exe = os.environ.get("LEROBOT_TRAIN", "lerobot-train")
    last = Path(out_dir) / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
    if last.exists():
        return [exe, f"--config_path={last}", "--resume=true"]
    if Path(out_dir).exists():
        # lerobot refuses to start in an existing directory; a leftover without a checkpoint is useless
        import shutil
        shutil.rmtree(out_dir)
    return [exe,
            f"--dataset.repo_id={params['dataset_repo_id']}",
            f"--policy.type={params['policy_type']}",
            f"--output_dir={out_dir}",
            f"--job_name=qortex_{params.get('policy_id', 'job')}",
            f"--policy.device={device}",
            f"--steps={int(params['steps'])}",
            f"--batch_size={int(params['batch_size'])}",
            f"--save_freq={int(params['save_freq'])}",
            "--policy.push_to_hub=false",
            "--wandb.enable=false",
            *os.environ.get("LEROBOT_EXTRA_ARGS", "").split(), *extra]


_NUM = r"([0-9]+(?:\.[0-9]+)?)([KMB]?)"
_STEP = re.compile(r"\bstep:\s*" + _NUM)
_LOSS = re.compile(r"\bloss:\s*([0-9.eE+-]+)")
_PARAMS = re.compile(r"num_total_params=\s*([0-9]+)")


def _num(m, i=1):
    return float(m.group(i)) * {"": 1, "K": 1e3, "M": 1e6, "B": 1e9}[m.group(i + 1)]


def parse_line(line, progress):
    """Pull step / loss / parameter count out of a lerobot log line into `progress`."""
    m = _STEP.search(line)
    if m:
        progress["step"] = int(_num(m))
    m = _LOSS.search(line)
    if m:
        try:
            progress["loss"] = float(m.group(1))
        except ValueError:
            pass
    m = _PARAMS.search(line)
    if m:
        progress["n_params"] = int(m.group(1))


# -- upload

def upload_to_hub(folder, policy_id):
    """Push the trained model folder to a private model repo. Returns (uri, meta)."""
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    user = os.environ.get("HF_USER") or api.whoami()["name"]
    repo = f"{user}/qortex-{policy_id.replace('_', '-')}"
    api.create_repo(repo, private=True, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=str(folder), repo_id=repo, repo_type="model", commit_message="Qortex training run")
    return f"hf://{repo}", {"repo_id": repo, "private": True, "host": "huggingface"}


# -- one job

def run_job(job, runs_dir=RUNS_DIR, uploader=upload_to_hub, device="cuda", on_checkpoint=None):
    """Train one claimed job to completion and report the outcome. Returns "done", "failed" or "cancelled"."""
    jid, p = job["id"], job["params"]
    out = Path(runs_dir) / jid
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_command(p, out, device)
    resumed = "--resume=true" in cmd
    print(f"[{jid}] {'resuming' if resumed else 'starting'}: {' '.join(cmd)}", flush=True)
    progress = {"step": 0, "total_steps": int(p["steps"]), "attempt": job.get("attempts", 1), "resumed": resumed}
    tail, state = [], {"cancelled": False, "stop": False}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def beat():
        while not state["stop"]:
            try:
                if call(f"/v1/worker/jobs/{jid}/heartbeat", {"progress": progress, "lease_seconds": LEASE_S}).get("cancelled"):
                    state["cancelled"] = True
                    proc.terminate()
                    return
            except Exception as e:  # noqa  a missed heartbeat is fine; the lease covers several
                print(f"[{jid}] heartbeat failed: {e}", flush=True)
            if on_checkpoint:
                on_checkpoint()
            for _ in range(int(float(os.environ.get("QORTEX_BEAT_S", BEAT_S)) * 10)):
                if state["stop"]:
                    return
                time.sleep(0.1)

    t = threading.Thread(target=beat, daemon=True)
    t.start()
    for line in proc.stdout:
        tail.append(line.rstrip())
        del tail[:-40]
        parse_line(line, progress)
        print(line, end="", flush=True)
    code = proc.wait()
    state["stop"] = True
    t.join(timeout=5)
    if on_checkpoint:
        on_checkpoint()
    if state["cancelled"]:
        print(f"[{jid}] cancelled by the server", flush=True)
        return "cancelled"
    model = out / "checkpoints" / "last" / "pretrained_model"
    if code != 0 or not model.exists():
        err = f"lerobot-train exited with code {code}" if code else "training finished but left no checkpoint"
        call(f"/v1/worker/jobs/{jid}/finish", {"error": err + ". Last output: " + " | ".join(tail[-6:])[-1500:]})
        return "failed"
    try:
        uri, meta = uploader(model, p.get("policy_id", jid))
    except Exception as e:  # noqa
        call(f"/v1/worker/jobs/{jid}/finish", {"error": f"training finished but the upload failed: {type(e).__name__}: {e}"})
        return "failed"
    size = sum(f.stat().st_size for f in model.rglob("*") if f.is_file())
    result = {"steps_done": progress.get("step") or int(p["steps"]), "final_loss": progress.get("loss"),
              "n_params": progress.get("n_params"), "size_bytes": size, "policy_type": p["policy_type"],
              "dataset_repo_id": p["dataset_repo_id"], "attempts": job.get("attempts", 1), "trainer": "lerobot"}
    call(f"/v1/worker/jobs/{jid}/finish", {"result": result, "artifact_uri": uri, "artifact_meta": meta})
    print(f"[{jid}] done: {uri}", flush=True)
    return "done"


def main(once=False):
    worker = os.environ.get("QORTEX_WORKER_NAME", "local-" + os.uname().nodename)
    print(f"gpu worker '{worker}' polling {_cfg()[0]} for {BACKENDS}", flush=True)
    while True:
        try:
            job = claim(worker)
        except (urllib.error.URLError, TimeoutError) as e:
            print("claim failed:", e, flush=True)
            job = None
        if job:
            run_job(job)
        if once:
            return
        if not job:
            time.sleep(30)


# -- Modal app (only defined when the modal package is present, i.e. under `modal deploy` / on Modal)

try:
    import modal
except ImportError:
    modal = None

if modal:
    app = modal.App("qortex-gpu-worker")
    runs = modal.Volume.from_name("qortex-runs", create_if_missing=True)
    secret = modal.Secret.from_name("qortex-worker")
    gpu_image = (modal.Image.debian_slim(python_version="3.12")
                 .apt_install("git", "ffmpeg")
                 .pip_install("lerobot", "huggingface_hub"))

    @app.function(image=gpu_image, gpu=os.environ.get("QORTEX_MODAL_GPU", "L4"), volumes={"/runs": runs}, secrets=[secret],
                  timeout=6 * 3600)
    def train(job: dict):
        # No Modal-level retries: if this container is preempted, the job's lease expires and the poller
        # hands it to a fresh container, which resumes from the checkpoint on the volume.
        return run_job(job, runs_dir=Path("/runs"), on_checkpoint=runs.commit)

    @app.function(image=modal.Image.debian_slim(python_version="3.12"), secrets=[secret], schedule=modal.Period(minutes=2))
    def poll():
        for _ in range(4):
            job = claim("modal")
            if not job:
                return
            print("starting GPU container for", job["id"])
            train.spawn(job)


if __name__ == "__main__":
    main(once="--once" in sys.argv)
