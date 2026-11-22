# Qortex

Model routing and a cloud-to-robot policy pipeline, in one API:
**route → fine-tune → distill → evaluate in sim → gated deploy → on-robot runtime → telemetry.**

"Qortex" is a working name.

```
server/        FastAPI control plane (API, inline worker, serves the docs site)
qortex_core/   sim tasks (MuJoCo), training, distillation, policy format, job runner
worker/        optional remote worker (run jobs on another machine)
sdk/           Python SDK + numpy-only on-robot runtime
docs/          docs site source (MkDocs Material)
examples/      sim_robot.py: a stand-in robot that runs the real runtime
tests/e2e.py   full pipeline test for arm, mobile base, drone and a custom robot
```

## Run it

```bash
pip install -r requirements.txt
python scripts/gen_api_docs.py && mkdocs build
uvicorn server.app:app --port 8000
python tests/e2e.py http://localhost:8000
```

Docs are then at http://localhost:8000, the interactive API reference at `/reference`.

## Deploy

`render.yaml` is a Render blueprint (Docker web service + 1 GB disk). See `docs/self-hosting.md`.
The Dockerfile has not been built in CI yet.

## What is real and what is not

Real and tested end to end: the router, sim data generation, fine-tuning, distillation with int8
quantization, sim evals, the deploy gate, rollback, hosted inference, the runtime, telemetry,
remote workers. All for the small built-in `qortex/mlp-bc` policy family.

Not built: training or serving VLA models (they are catalogued and routed, and their jobs wait
for a GPU backend), custom sim twins, a dashboard, billing, hardened auth. Third-party model
specs in `server/catalog.json` are approximate and unverified. See `docs/limits.md`.
