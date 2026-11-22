# Self-hosting and workers

Qortex is one container: the API, an inline job worker and this docs site.

## Run locally

```bash
pip install -r requirements.txt
mkdocs build                                   # builds the docs into ./site
uvicorn server.app:app --port 8000
python tests/e2e.py http://localhost:8000      # full pipeline, about 45 seconds
```

## Run with Docker

```bash
docker build -t qortex .
docker run -p 8000:8000 -v qortex-data:/data qortex
```

## Deploy on Render

The repo includes `render.yaml`. In Render choose **New > Blueprint**, pick the repo, and deploy. It creates one web service with a 1 GB disk mounted at `/data`. Then add your custom domain in the service's settings and point a CNAME at the address Render shows.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `QORTEX_DATA` | `./data` | Directory for the SQLite database and artifacts. Put it on a persistent disk. |
| `QORTEX_INLINE_WORKER` | `1` | Run jobs on a thread inside the API process |
| `QORTEX_WORKER_TOKEN` | unset | Shared secret that lets remote workers claim jobs |
| `QORTEX_SIGNUP` | `1` | Allow anyone to create an API key with `POST /v1/keys`. Set to `0` after creating yours. |

## Remote workers

A remote worker claims queued jobs over HTTPS, runs them and posts the result. Use one to move training off the API host, for example onto a GPU machine.

```bash
QORTEX_URL=https://YOUR-HOST QORTEX_WORKER_TOKEN=... python -m worker.remote
```

Set `QORTEX_INLINE_WORKER=0` on the server if remote workers should take every job.

### Adding a model backend

Jobs carry a `backend` name. Built-in policies use `native`. A fine-tune request for a catalog model such as `lerobot/smolvla` is queued with backend `gpu:lerobot/smolvla` and waits. To serve it:

1. Add the backend name to `BACKENDS` in `qortex_core/jobs.py` on the worker machine.
2. Add a branch in `jobs.run()` that trains the model and returns an artifact.
3. Teach the runtime to load that artifact format.

None of the VLA backends exist yet.

## Security notes for this version

- API keys are stored in plain text in SQLite.
- There is no rate limiting and no per-key quota.
- Signup is open by default.

Do not expose this version to the public internet with data you care about.
