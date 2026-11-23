"""Thin HTTP client for the Qortex API. Standard library only."""
import json
import os
import time
import urllib.error
import urllib.request


DEFAULT_URL = "https://qortex.quantumroboticslab.ai"


class QortexError(Exception):
    def __init__(self, status, message):
        super().__init__(f"{status}: {message}")
        self.status, self.message = status, message


class Client:
    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key or os.environ.get("QORTEX_API_KEY", "")
        self.base_url = (base_url or os.environ.get("QORTEX_URL", DEFAULT_URL)).rstrip("/")

    # -- transport
    def _req(self, method, path, body=None, raw=False, timeout=90):
        req = urllib.request.Request(self.base_url + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
                return data if raw else json.loads(data)
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read()).get("error", e.reason)
            except Exception:  # noqa
                msg = e.reason
            raise QortexError(e.code, msg) from None

    def get(self, path):
        return self._req("GET", path)

    def post(self, path, body=None):
        return self._req("POST", path, body or {})

    # -- auth
    @classmethod
    def signup(cls, email, base_url=None):
        c = cls("", base_url)
        c.api_key = c.post("/v1/keys", {"email": email})["api_key"]
        return c

    # -- router
    def models(self):
        return self.get("/v1/models")["models"]

    def recommend(self, robot_id=None, robot=None, task=None, constraints=None):
        return self.post("/v1/recommend", {"robot_id": robot_id, "robot": robot,
                                           "task": task or {}, "constraints": constraints or {}})

    # -- objects
    def create_robot(self, name, type, compute=None, **kw):
        return self.post("/v1/robots", dict(name=name, type=type, compute=compute, **kw))

    def create_dataset(self, robot_id, name="dataset"):
        return self.post("/v1/datasets", {"robot_id": robot_id, "name": name})

    def upload_episodes(self, dataset_id, episodes):
        """episodes: list of {"observations": [[...]], "actions": [[...]]}, actions in [-1, 1]."""
        return self.post(f"/v1/datasets/{dataset_id}/episodes", {"episodes": episodes})

    def generate(self, dataset_id, episodes=150, noise=0.1, wait=True):
        d = self.post(f"/v1/datasets/{dataset_id}/generate", {"episodes": episodes, "noise": noise})
        if wait:
            self.wait(d["job_id"])
            d = self.get(f"/v1/datasets/{dataset_id}")
        return d

    def finetune(self, robot_id, dataset_id, base_model="qortex/mlp-bc", wait=True, **kw):
        p = self.post("/v1/policies", dict(robot_id=robot_id, dataset_id=dataset_id, base_model=base_model, **kw))
        return self._wait_policy(p) if wait and p["status"] == "queued" else p

    def distill(self, policy_id, hidden=(32, 32), quantize="int8", wait=True, **kw):
        p = self.post(f"/v1/policies/{policy_id}/distill", dict(hidden=list(hidden), quantize=quantize, **kw))
        return self._wait_policy(p) if wait else p

    def _wait_policy(self, p):
        self.wait(p["job_id"])
        p = self.get(f"/v1/policies/{p['id']}")
        end = time.time() + 300          # auto-eval runs right after training
        while p.get("auto_eval") and not p.get("latest_eval") and time.time() < end:
            time.sleep(0.5)
            p = self.get(f"/v1/policies/{p['id']}")
        return p

    def policy(self, policy_id):
        return self.get(f"/v1/policies/{policy_id}")

    def evaluate(self, policy_id, episodes=50, wait=True):
        e = self.post("/v1/evals", {"policy_id": policy_id, "episodes": episodes})
        if wait:
            self.wait(e["job_id"])
            e = self.get(f"/v1/evals/{e['id']}")
        return e

    def deploy(self, policy_id, robot_id, target="on_robot", min_success_rate=0.8, force=False, max_action_mse=None):
        return self.post("/v1/deployments", {"policy_id": policy_id, "robot_id": robot_id, "target": target, "force": force,
                                             "gate": {"min_success_rate": min_success_rate, "max_action_mse": max_action_mse}})

    def rollback(self, robot_id):
        return self.post(f"/v1/robots/{robot_id}/rollback")

    def current_deployment(self, robot_id):
        return self.get(f"/v1/robots/{robot_id}/deployment")

    def download_artifact(self, policy_id):
        return self._req("GET", f"/v1/policies/{policy_id}/artifact", raw=True)

    def act(self, policy_id, observation):
        return self.post(f"/v1/policies/{policy_id}/act", {"observation": [float(x) for x in observation]})["action"]

    def telemetry(self, robot_id, **stats):
        return self.post(f"/v1/robots/{robot_id}/telemetry", stats)

    # -- jobs
    def wait(self, job_id, timeout=600):
        end = time.time() + timeout
        while time.time() < end:
            j = self.get(f"/v1/jobs/{job_id}?wait=30")
            if j["status"] == "done":
                return j
            if j["status"] == "failed":
                raise QortexError(500, f"job {job_id} failed: {j['error']}")
        raise QortexError(504, f"job {job_id} did not finish in {timeout}s")
