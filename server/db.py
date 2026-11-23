"""Storage. SQLite by default, Postgres when DATABASE_URL (or POSTGRES_URL) is set.

Two tables: obj (JSON documents) and blob (datasets and policy artifacts).
Everything lives in the database, so the app needs no local disk and can run
on serverless hosts.
"""
import json
import os
import secrets
import sqlite3
import threading
import time

PG_URL = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL") or ""
_lock = threading.RLock()
_conn = None


def _connect():
    global _conn
    if PG_URL:
        import psycopg
        _conn = psycopg.connect(PG_URL, autocommit=True)
        blob_t = "BYTEA"
    else:
        data = os.environ.get("QORTEX_DATA", "./data")
        os.makedirs(data, exist_ok=True)
        _conn = sqlite3.connect(os.path.join(data, "qortex.db"), check_same_thread=False, isolation_level=None)
        blob_t = "BLOB"
    for stmt in (
        "CREATE TABLE IF NOT EXISTS obj (id TEXT PRIMARY KEY, kind TEXT, owner TEXT, created DOUBLE PRECISION, data TEXT)",
        "CREATE INDEX IF NOT EXISTS obj_kind ON obj (kind, owner, created)",
        f"CREATE TABLE IF NOT EXISTS blob (id TEXT PRIMARY KEY, data {blob_t})",
    ):
        _conn.execute(stmt)


def q(sql, params=()):
    """Run one statement and return all rows. Reconnects once if the connection dropped."""
    global _conn
    with _lock:
        for attempt in (0, 1):
            try:
                if _conn is None:
                    _connect()
                cur = _conn.execute(sql.replace("?", "%s") if PG_URL else sql, params)
                return cur.fetchall() if cur.description else []
            except Exception as e:  # noqa
                dropped = type(e).__name__ in ("OperationalError", "InterfaceError") and PG_URL
                if attempt or not dropped:
                    raise
                _conn = None


class NotFound(Exception):
    pass


def new_id(prefix):
    return f"{prefix}_{secrets.token_hex(6)}"


def put(kind, owner, data, id=None):
    id = id or new_id(kind[:3])
    data = dict(data, id=id, created=data.get("created", round(time.time(), 3)))
    q("INSERT INTO obj (id, kind, owner, created, data) VALUES (?,?,?,?,?) "
      "ON CONFLICT (id) DO UPDATE SET data = excluded.data", (id, kind, owner, data["created"], json.dumps(data)))
    return data


def get(kind, id, owner=None):
    rows = q("SELECT owner, data FROM obj WHERE id=? AND kind=?", (id, kind))
    if not rows or (owner is not None and rows[0][0] != owner):
        raise NotFound(f"{kind} '{id}' not found")
    return json.loads(rows[0][1])


def owner_of(id):
    return q("SELECT owner FROM obj WHERE id=?", (id,))[0][0]


def update(kind, id, **fields):
    with _lock:
        data = dict(get(kind, id), **fields)
        q("UPDATE obj SET data=? WHERE id=?", (json.dumps(data), id))
    return data


def listing(kind, owner, **where):
    rows = q("SELECT data FROM obj WHERE kind=? AND owner=? ORDER BY created DESC", (kind, owner))
    out = [json.loads(r[0]) for r in rows]
    return [o for o in out if all(o.get(k) == v for k, v in where.items())]


def queued_jobs():
    rows = q("SELECT data FROM obj WHERE kind='job' AND data LIKE ? ORDER BY created", ('%"status": "queued"%',))
    return [j for j in (json.loads(r[0]) for r in rows) if j["status"] == "queued"]


def write_blob(id, b):
    q("INSERT INTO blob (id, data) VALUES (?,?) ON CONFLICT (id) DO UPDATE SET data = excluded.data", (id, b))


def read_blob(id):
    rows = q("SELECT data FROM blob WHERE id=?", (id,))
    if not rows:
        raise NotFound(f"blob '{id}' not found")
    return bytes(rows[0][0])
