# Training ACT on a GPU

`qortex/mlp-bc` trains inside the API in seconds because it is tiny and state-only. Camera-based policies are not:
ACT needs a GPU for an hour or two. Qortex does not own GPUs, so you connect one. The API queues the job, a **GPU
worker** on your side picks it up, trains with [LeRobot](https://github.com/huggingface/lerobot), pushes the weights
to a private Hugging Face repo, and reports back.

!!! warning "Status: built and tested against a stand-in trainer, not yet run on a real GPU"
    The job protocol (lease, heartbeat, crash, resume, cancel, artifact pointer, deploy gate) is covered by the
    automated tests using a fake `lerobot-train`. The real training command and the Modal image have not been
    executed on a GPU yet. Expect to fix small things on the first run.

## What you need

| | |
|---|---|
| A dataset in LeRobot format on the Hugging Face Hub | Record with `lerobot-record`, or start with a public one such as `lerobot/pusht` |
| A Hugging Face token with write access | Weights go to a private model repo under your account |
| The worker token | The `QORTEX_WORKER_TOKEN` set on the Qortex server |
| A GPU | Modal (about $0.80/hour for an L4, billed per second) or your own machine |

## 1. Queue a training job

```python
from qortex import Client
c = Client("qx_...")

robot = c.create_robot("so101", "custom", obs_dim=6, action_dim=6, control_hz=30)
ds = c.create_dataset(robot["id"], hub_repo_id="your-hf-user/your-dataset")   # a reference, nothing is uploaded
policy = c.finetune(robot["id"], ds["id"], base_model="lerobot/act", steps=20000, batch_size=8, wait=False)
print(policy["status"], policy["message"])     # waiting_for_gpu_worker
```

For a public dataset Qortex reads its `meta/info.json` and rejects a mismatch between the dataset's state and action
sizes and the robot's. For a private dataset pass `private=True`; the check is skipped and the worker's token must be
able to read it.

## 2a. Run the worker on Modal

One-time setup, from any machine with Python:

```bash
pip install modal && modal setup
modal secret create qortex-worker QORTEX_WORKER_TOKEN=... HF_TOKEN=hf_... QORTEX_URL=https://qortex.quantumroboticslab.ai
modal deploy worker/gpu.py
```

Or without a laptop: add `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `QORTEX_WORKER_TOKEN` and `HF_TOKEN` as repository
secrets on GitHub and run the **Deploy GPU worker** workflow from the Actions tab.

After that nothing runs while the queue is empty. A small CPU function checks for jobs every two minutes and starts
one GPU container per job. Set `QORTEX_MODAL_GPU=A10G` (or another type) before `modal deploy` to change the GPU.

## 2b. Run the worker on your own GPU

```bash
pip install lerobot huggingface_hub
export QORTEX_WORKER_TOKEN=... HF_TOKEN=hf_...
python -m worker.gpu            # or: python -m worker.gpu --once
```

On RTX 50-series cards add `LEROBOT_EXTRA_ARGS="--dataset.video_backend=pyav"` if video decoding fails.

## 3. Watch it and use the result

```python
p = c.get(f"/v1/policies/{policy['id']}")
p["status"]        # waiting_for_gpu_worker -> training -> ready (or failed, with the trainer's last output)
p["progress"]      # {"step": 6000, "total_steps": 20000, "loss": 0.083, "attempt": 1, "resumed": false}
p["artifact_uri"]  # hf://your-hf-user/qortex-pol-...
```

`GET /healthz` lists `gpu_workers` seen in the last ten minutes, and the router marks `lerobot/act` as runnable while
one is connected.

## If the GPU disappears mid-run

The worker holds a ten-minute lease on the job and renews it every 30 seconds. Checkpoints are written to a
persistent directory keyed by job id. If the machine is preempted or loses power, the lease runs out, the job is
handed to the next worker that asks, and training resumes from the last checkpoint. A trainer crash is different: it
is reported as `failed` with the last lines of output, and is not retried.

Deleting the robot (or the account) cancels its running jobs: the next heartbeat tells the worker to stop.

## What a GPU-trained policy can and cannot do here

| | |
|---|---|
| Versioned as a policy, with training metrics and lineage | Yes |
| Deploy record, rollback, telemetry | Yes |
| Evaluation | **No.** The scorecard is training loss only, marked `validated: false` |
| Deploy gate | Always blocks. Deploying needs `force=True`, recorded as `forced` |
| Hosted inference (`/act`), artifact download, distillation | No. The weights live on the Hub; load them with LeRobot |
| Qortex on-robot runtime | No. It raises a clear error; run the policy with LeRobot on the robot |

Closing the evaluation gap (a held-out action error for chunked policies, then sim) is the next piece of work here.
