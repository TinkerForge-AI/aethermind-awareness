# aethermind-awareness/datastore/bridge.py
from __future__ import annotations
import os, json, importlib
from typing import Any, Dict, Iterator, List, Optional
from dotenv import load_dotenv

load_dotenv()

# -------- DB connector (DuckDB) ------------------------------------
import duckdb

# If you already expose a connect() in aethermind-interpretation that returns a DuckDB
# connection, you can point to it via env AETHERMIND_INTERPRETATION_CONNECT="module.attr".
INTERP_CONNECT_SPEC = os.environ.get("AETHERMIND_INTERPRETATION_CONNECT", "")

DB_PATH = os.environ.get("AETHERMIND_DB_PATH", "aethermind.duckdb")

def _import_connect() -> Optional[Any]:
    if not INTERP_CONNECT_SPEC:
        return None
    try:
        mod_name, attr = INTERP_CONNECT_SPEC.rsplit(".", 1)
        return getattr(importlib.import_module(mod_name), attr)
    except Exception:
        return None

_connect_from_interp = _import_connect()

if _connect_from_interp is not None:
    def connect(read_only: bool = False):
        return _connect_from_interp(read_only=read_only)
else:
    def connect(read_only: bool = False):
        return duckdb.connect(DB_PATH, read_only=read_only)

# -------- Optional NumPy for vector blobs --------------------------
try:
    import numpy as np
except Exception:
    np = None  # type: ignore

MEDIA_ROOT = os.environ.get("AETHERMIND_MEDIA_ROOT", "")

# -------- helpers ---------------------------------------------------
def _ensure_aux_tables(con, read_only: bool = False) -> None:
    """Create lightweight tables for agency annotations/runs if they don’t exist."""
    if read_only:
        return  # Skip table creation in read-only mode

    con.execute("""
        CREATE TABLE IF NOT EXISTS agent_feedback (
            episode_id TEXT NOT NULL,
            label      TEXT NOT NULL,        -- 'keep' | 'skip' | custom
            note       TEXT,
            created_ts TIMESTAMP DEFAULT (strftime('%s','now')),
            PRIMARY KEY (episode_id, created_ts)
        );
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS agent_runs (
            run_id     TEXT PRIMARY KEY,
            goal       TEXT,
            config_json TEXT,
            started_ts TIMESTAMP DEFAULT (strftime('%s','now')),
            ended_ts   TIMESTAMP,
            outcome    TEXT,
            details_json TEXT
        );
    """)

def _rowdicts(rows, description) -> List[Dict[str, Any]]:
    cols = [d[0] for d in description]
    return [{cols[i]: r[i] for i in range(len(cols))} for r in rows]

def _json_or_none(v: Any) -> Any:
    if v is None: return None
    if isinstance(v, (dict, list)): return v
    try:
        return json.loads(v)
    except Exception:
        return v

def _maybe_vec(buf: Optional[bytes]):
    if buf is None: return None
    if np is None:  return buf
    arr = np.frombuffer(buf, dtype=np.float32)
    return arr.copy()

def _resolve_media_path(p: Optional[str]) -> Optional[str]:
    if not p: return p
    if os.path.isabs(p) and os.path.exists(p):
        return p
    if MEDIA_ROOT:
        cand = os.path.normpath(os.path.join(MEDIA_ROOT, p))
        if os.path.exists(cand): return cand
    ap = os.path.abspath(p)
    return ap if os.path.exists(ap) else p

# -------- public API ------------------------------------------------
def find_event_by_uid(event_uid: str) -> Optional[str]:
    con = connect(read_only=True)
    row = con.execute("SELECT event_id FROM events WHERE event_uid = ?", [event_uid]).fetchone()
    return row[0] if row else None

def fetch_events(
    session_id: Optional[str] = None,
    since_ts: Optional[float] = None,
    until_ts: Optional[float] = None,
    limit: Optional[int] = None,
    include_vectors: bool = True,
) -> List[Dict[str, Any]]:
    con = connect(read_only=True)
    _ensure_aux_tables(con, read_only=True)  # Pass read_only flag

    where, args = [], []
    if session_id:
        where.append("session_id = ?"); args.append(session_id)
    if since_ts is not None:
        where.append("start_ts >= ?"); args.append(float(since_ts))
    if until_ts is not None:
        where.append("end_ts <= ?"); args.append(float(until_ts))

    veccol = "embeddings_F" if include_vectors else "CAST(NULL AS BLOB) AS embeddings_F"
    sql = f"""
        SELECT event_id, session_id, start_ts, end_ts,
               video_path, audio_path, scene_type, raw, {veccol}
        FROM events
    """
    if where: sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY start_ts ASC"
    if limit: sql += f" LIMIT {int(limit)}"

    cur = con.execute(sql, args)
    rows = cur.fetchall()
    evs = _rowdicts(rows, cur.description)

    out: List[Dict[str, Any]] = []
    for e in evs:
        raw = _json_or_none(e.get("raw"))
        out.append({
            "event_id": e["event_id"],
            "session_id": e["session_id"],
            "start": float(e["start_ts"]) if e["start_ts"] is not None else None,
            "end": float(e["end_ts"]) if e["end_ts"] is not None else None,
            "duration": (float(e["end_ts"]) - float(e["start_ts"]))
                        if (e["start_ts"] is not None and e["end_ts"] is not None) else None,
            "scene_type": e.get("scene_type") or "",
            "video_path": _resolve_media_path(e.get("video_path")),
            "audio_path": _resolve_media_path(e.get("audio_path")),
            "raw": raw,
            "F": _maybe_vec(e.get("embeddings_F")),
        })
    return out

def fetch_episodes(
    filters: Optional[Dict[str, Any]] = None,
    order_by: str = "salience_mean DESC, start_ts ASC",
    limit: Optional[int] = None,
    include_vectors: bool = True,
) -> List[Dict[str, Any]]:
    filters = filters or {}
    con = connect(read_only=True)
    _ensure_aux_tables(con, read_only=True)  # Pass read_only flag

    where, args = [], []

    if "session_id" in filters:
        where.append("""
            EXISTS (
              SELECT 1
              FROM episode_events ee
              JOIN events ev ON ev.event_id = ee.event_id
              WHERE ee.episode_id = episodes.episode_id
                AND ev.session_id = ?
            )
        """)
        args.append(filters["session_id"])

    if "min_events" in filters:
        where.append("COALESCE(num_events,0) >= ?")
        args.append(int(filters["min_events"]))

    if "since_ts" in filters:
        where.append("end_ts >= ?")
        args.append(float(filters["since_ts"]))

    if "until_ts" in filters:
        where.append("start_ts <= ?")
        args.append(float(filters["until_ts"]))

    if "scene_type" in filters:
        where.append("caption ILIKE ?")
        args.append(f"%{filters['scene_type']}%")

    veccol = "f_embed" if include_vectors else "CAST(NULL AS BLOB) AS f_embed"
    sql = f"""
        SELECT episode_id, start_ts, end_ts, caption, tags_json,
               {veccol},
               summary, thought_text, valence_guess, confidence,
               COALESCE(salience_mean,0.0) AS salience_mean,
               COALESCE(num_events,0) AS num_events
        FROM episodes
    """
    if where: sql += " WHERE " + " AND ".join(where)
    if order_by: sql += f" ORDER BY {order_by}"
    if limit: sql += f" LIMIT {int(limit)}"

    cur = con.execute(sql, args)
    rows = cur.fetchall()
    eps = _rowdicts(rows, cur.description)

    out: List[Dict[str, Any]] = []
    for ep in eps:
        out.append({
            "episode_id": ep["episode_id"],
            "start": float(ep["start_ts"]) if ep["start_ts"] is not None else None,
            "end": float(ep["end_ts"]) if ep["end_ts"] is not None else None,
            "duration": (float(ep["end_ts"]) - float(ep["start_ts"]))
                        if (ep["start_ts"] is not None and ep["end_ts"] is not None) else None,
            "caption": ep.get("caption") or "",
            "tags": _json_or_none(ep.get("tags_json")) or [],
            "F": _maybe_vec(ep.get("f_embed")),
            "summary": ep.get("summary") or "",
            "thought_text": ep.get("thought_text") or "",
            "valence": ep.get("valence_guess"),
            "confidence": float(ep["confidence"]) if ep.get("confidence") is not None else None,
            "salience_mean": float(ep.get("salience_mean") or 0.0),
            "num_events": int(ep.get("num_events") or 0),
        })
    return out

def fetch_episode_members(episode_id: str, include_vectors: bool = False) -> List[Dict[str, Any]]:
    con = connect(read_only=True)
    veccol = "embeddings_F" if include_vectors else "CAST(NULL AS BLOB) AS embeddings_F"
    sql = f"""
        SELECT ev.event_id, ev.session_id, ev.start_ts, ev.end_ts,
               ev.video_path, ev.audio_path, ev.scene_type, ev.raw, {veccol}
        FROM episode_events ee
        JOIN events ev ON ev.event_id = ee.event_id
        WHERE ee.episode_id = ?
        ORDER BY ee.ord ASC
    """
    cur = con.execute(sql, [episode_id])
    rows = cur.fetchall()
    evs = _rowdicts(rows, cur.description)
    out = []
    for e in evs:
        out.append({
            "event_id": e["event_id"],
            "session_id": e["session_id"],
            "start": float(e["start_ts"]) if e["start_ts"] is not None else None,
            "end": float(e["end_ts"]) if e["end_ts"] is not None else None,
            "video_path": _resolve_media_path(e.get("video_path")),
            "audio_path": _resolve_media_path(e.get("audio_path")),
            "scene_type": e.get("scene_type") or "",
            "raw": _json_or_none(e.get("raw")),
            "F": _maybe_vec(e.get("embeddings_F")),
        })
    return out

def top_episodes_for_stitching(min_events: int = 3, k: int = 10) -> List[str]:
    con = connect(read_only=True)
    rows = con.execute("""
        SELECT episode_id
        FROM episodes
        WHERE COALESCE(num_events,0) >= ?
        ORDER BY COALESCE(salience_mean,0.0) DESC, start_ts ASC
        LIMIT ?
    """, [int(min_events), int(k)]).fetchall()
    return [r[0] for r in rows]

def iter_training_windows(
    session_id: str,
    window_sec: float = 10.0,
    stride_sec: float = 2.0,
    include_vectors: bool = True,
) -> Iterator[Dict[str, Any]]:
    events = fetch_events(session_id=session_id, include_vectors=include_vectors)
    if not events:
        return
    events.sort(key=lambda e: e["start"] or 0.0)
    t0 = float(events[0]["start"] or 0.0)
    tN = float(events[-1]["end"] or events[-1]["start"] or t0)

    cur = t0
    while cur < tN:
        w_start, w_end = cur, cur + window_sec
        bucket, vecs = [], []
        for e in events:
            es = float(e["start"] or 0.0)
            ee = float(e["end"] or es)
            if ee < w_start or es > w_end:
                continue
            bucket.append(e)
            if include_vectors and np is not None and e.get("F") is not None:
                vecs.append(e["F"])
        fused_mean = None
        if include_vectors and np is not None and vecs:
            try:
                fused_mean = np.stack(vecs, axis=0).mean(axis=0)
            except Exception:
                fused_mean = None
        yield {
            "session_id": session_id,
            "start": w_start,
            "end": w_end,
            "duration": window_sec,
            "events": bucket,
            "fused_embedding_mean": fused_mean,
        }
        cur += stride_sec

def record_feedback(episode_id: str, label: str, note: Optional[str] = None) -> None:
    con = connect(read_only=False)
    _ensure_aux_tables(con)
    con.execute(
        "INSERT INTO agent_feedback (episode_id, label, note) VALUES (?, ?, ?)",
        [episode_id, label, note]
    )

def log_agent_run(
    run_id: str,
    goal: str,
    outcome: str,
    config: Optional[Dict[str, Any]] = None,
    details: Optional[Dict[str, Any]] = None,
    ended_ts: Optional[float] = None,
) -> None:
    """
    DuckDB: use INSERT OR REPLACE for simple upsert by PRIMARY KEY(run_id).
    (Replaces the entire row; fine for run log.)
    """
    con = connect(read_only=False)
    _ensure_aux_tables(con)
    cfg_json = json.dumps(config or {})
    det_json = json.dumps(details or {})
    # ended_ts: write as TIMESTAMP if provided (seconds since epoch -> to_timestamp)
    ended_ts_sql = "to_timestamp(?)" if ended_ts is not None else "NULL"
    params = [run_id, goal, cfg_json, outcome, det_json]
    if ended_ts is not None:
        params.append(float(ended_ts))
    con.execute(f"""
        INSERT OR REPLACE INTO agent_runs
          (run_id, goal, config_json, outcome, details_json, ended_ts)
        VALUES (?, ?, ?, ?, ?, {ended_ts_sql})
    """, params)
