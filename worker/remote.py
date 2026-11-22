"""Remote worker. Run this on any machine that should execute jobs,
for example a GPU box:

    QORTEX_URL=https://your-host QORTEX_WORKER_TOKEN=... python -m worker.remote

It claims queued jobs, runs them with qortex_core.jobs, and posts results.
Add a backend name to BACKENDS and a branch in jobs.run() to support a
new model family.
"""
import base64
import json
import os
import time
import traceback
import urllib.request

from qortex_core import jobs

URL = os.environ.get("QORTEX_URL", "http://localhost:8000").rstrip("/")
TOKEN = os.environ["QORTEX_WORKER_TOKEN"]


def call(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "X-Worker-Token": TOKEN})
    return json.load(urllib.request.urlopen(req, timeout=120))


def main(once=False):
    print(f"worker polling {URL} for backends {jobs.BACKENDS}")
    while True:
        job = call("/v1/worker/claim", {"backends": jobs.BACKENDS})["job"]
        if not job:
            if once:
                return
            time.sleep(1)
            continue
        print("claimed", job["id"], job["kind"])
        try:
            blobs = {k: base64.b64decode(v) for k, v in job["blobs"].items()}
            result, art = jobs.run(job["kind"], job["params"], blobs)
            call(f"/v1/worker/jobs/{job['id']}/finish",
                 {"result": result, "artifact_b64": base64.b64encode(art).decode() if art else None})
        except Exception as e:  # noqa
            traceback.print_exc()
            call(f"/v1/worker/jobs/{job['id']}/finish", {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
