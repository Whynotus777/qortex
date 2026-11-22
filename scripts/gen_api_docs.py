"""Generate docs/api.md from the live OpenAPI schema so it cannot drift."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("QORTEX_DATA", "/tmp/qortex-docs")
from server.app import app

spec = app.openapi()
schemas = spec["components"]["schemas"]

def fields(ref):
    s = schemas[ref.split("/")[-1]]
    rows = []
    for name, f in s.get("properties", {}).items():
        t = f.get("type") or ("object" if "$ref" in f else "/".join(a.get("type", "object") for a in f.get("anyOf", [])) or "any")
        if "$ref" in f or any("$ref" in a for a in f.get("allOf", [])):
            sub = f.get("$ref") or f["allOf"][0]["$ref"]
            t = "object: " + ", ".join(f"`{k}`" for k in schemas[sub.split('/')[-1]].get("properties", {}))
        req = "yes" if name in s.get("required", []) else "no"
        d = f.get("default", "")
        rows.append(f"| `{name}` | {t} | {req} | {'' if d in ('', None) else '`'+str(d)+'`'} | {f.get('description', '')} |")
    return rows

out = ["# API reference", "",
       "Base URL: your server, for example `https://YOUR-HOST`. All endpoints are under `/v1`.", "",
       "An interactive version of this page, generated from the same schema, is at [`/reference`](/reference).", "",
       "## Authentication", "",
       "Send your key as a bearer token on every request except `POST /v1/keys`, `GET /v1/models` and `GET /v1/devices`.", "",
       "```bash", "curl $QORTEX_URL/v1/robots -H \"Authorization: Bearer $QORTEX_API_KEY\"", "```", "",
       "## Errors", "", "Errors return JSON `{\"error\": \"message\"}` with these status codes:", "",
       "| Code | Meaning |", "|---|---|",
       "| 401 | Missing or invalid API key |", "| 404 | Object not found, or it belongs to another key |",
       "| 409 | Object is not in the right state yet, for example a policy still training |",
       "| 412 | Deployment blocked by the eval gate |", "| 422 | Invalid request body |", ""]
tags = {}
for path, methods in spec["paths"].items():
    for method, op in methods.items():
        tags.setdefault(op.get("tags", ["Other"])[0], []).append((method.upper(), path, op))
for tag, ops in tags.items():
    out += [f"## {tag}", ""]
    for method, path, op in ops:
        out += [f"### {op['summary']}", "", f"`{method} {path}`", ""]
        q = [p for p in op.get("parameters", []) if p["in"] == "query"]
        if q:
            out += ["Query parameters: " + ", ".join(f"`{p['name']}`" for p in q) + ".", ""]
        body = op.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema", {})
        if "$ref" in body:
            out += ["| Field | Type | Required | Default | Description |", "|---|---|---|---|---|", *fields(body["$ref"]), ""]
open(os.path.join(os.path.dirname(__file__), "..", "docs", "api.md"), "w").write("\n".join(out) + "\n")
print("wrote docs/api.md", sum(len(v) for v in tags.values()), "endpoints")
