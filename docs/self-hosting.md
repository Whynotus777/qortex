# Self-hosting and workers

Qortex is one FastAPI app. It serves the API and this docs site, and it runs in two modes:

| Mode | Storage | Jobs | Use it for |
|---|---|---|---|
| **Serverless** (Vercel) | Postgres | Run inside the request that creates them | The hosted deployment |
| **Container** (Docker, a VM, your laptop) | SQLite or Postgres | Background worker thread, or remote workers | Local development, GPU machines |

## Deploy on Vercel

The repo is set up for Vercel: `pyproject.toml` names the entrypoint and the docs build command, and `vercel.json` sets a 300 second function limit.

1. In Vercel choose **Add New > Project** and import the repo. Keep the defaults and deploy.
2. In the project open **Storage**, create a Postgres database (Neon) and connect it to the project. This sets `DATABASE_URL`.
3. Redeploy so the function picks up the database.
4. In **Settings > Domains** add your subdomain.

Without a database the deployment still builds, but nothing is saved between requests.

On Vercel each training, distillation and eval job runs inside its HTTP request. The built-in jobs take a few seconds. Jobs that need a GPU cannot run there; they wait for a [remote worker](#remote-workers).

## Run locally

```bash
pip install -r requirements.txt
python scripts/build_docs.py                   # builds the docs into ./public
uvicorn server.app:app --port 8000
python tests/e2e.py http://localhost:8000      # full pipeline, about 45 seconds
```

## Run with Docker

```bash
docker build -t qortex .
docker run -p 8000:8000 -v qortex-data:/data qortex
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | unset | Postgres connection string. When set, all data and artifacts are stored in Postgres. `POSTGRES_URL` also works. |
| `QORTEX_DATA` | `./data` | Directory for the SQLite file, used only when no Postgres URL is set |
| `QORTEX_SYNC_JOBS` | `1` on Vercel, else `0` | Run jobs inside the request instead of on a background worker |
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

- Only a hash of each API key is stored, so a lost key cannot be recovered. Create a new one.
- There is no rate limiting and no per-key quota.
- Signup is open by default, and every key can start training jobs on your server. Set `QORTEX_SIGNUP=0` once you have your keys.

Do not expose this version to the public internet with data you care about.
