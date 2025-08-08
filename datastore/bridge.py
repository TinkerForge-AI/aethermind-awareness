# fetch_events(session_id|since|limit)
# fetch_episodes(filters, order_by='salience_mean', limit)
# iter_training_windows(window_sec=10, stride=2) → returns sequences with {video_paths, audio_paths, actions, text, tags, F-embed}
# record_feedback(episode_id, keep|skip, note)
# log_agent_run(goal, actions, outcomes)

# aethermind-agency/datastore/bridge.py
from __future__ import annotations
import os
import json
import duckdb
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


# Use .env for all relevant paths. Try to import connect from aethermind-interpretation, else fallback to local duckdb using env vars.
from dotenv import load_dotenv
load_dotenv()

DB_MODULE_PATH = os.environ.get("AETHERMIND_INTERPRETATION_PATH", "memory.db")
DB_PATH = os.environ.get("AETHERMIND_DB_PATH", "aethermind.db")

try:
    # Dynamically import connect from the specified module path
    import importlib
    module_name, attr = DB_MODULE_PATH.rsplit('.', 1)
    connect = getattr(importlib.import_module(module_name), attr)
except Exception as e:
    # Fallback: define a simple connect() using duckdb for local testing
    def connect(read_only: bool = False):
        db_path = DB_PATH
        uri = f"file:{db_path}?mode=ro" if read_only else db_path
        return duckdb.connect(uri)

try:
    import numpy as np
except Exception:  # numpy optional; we’ll degrade gracefully
    np = None  # type: ignore

MEDIA_ROOT = os.environ.get("AETHERMIND_MEDIA_ROOT", "")


# ----------------------------- utils -----------------------------

def _ensure_aux_tables(con) -> None:
    """Create lightweight tables for agency annotations/runs if they don’t exist."""
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
    out = []
    for r in rows:
        out.append({c: r[i] for i, c in enumerate(cols)})
    return out


def _json_or_none(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(v)
    except Exception:
        return v  # already native or not JSON


def _maybe_vec(buf: Optional[bytes]):
    """Convert BLOB → np.float32 array if NumPy available; else return as-is."""
    if buf is None:
        return None
    if np is None:
        return buf
    arr = np.frombuffer(buf, dtype=np.float32)
    return arr.copy()  # writable


def _resolve_media_path(p: Optional[str]) -> Optional[str]:
    """Prefer MEDIA_ROOT/p if it exists, else absolute(p) if exists, else return p."""
    if not p:
        return p
    # absolute path exists?
    if os.path.isabs(p) and os.path.exists(p):
        return p
    # MEDIA_ROOT join
    if MEDIA_ROOT:
        cand = os.path.normpath(os.path.join(MEDIA_ROOT, p))
        if os.path.exists(cand):
            return cand
    # last resort: try abspath of p relative to CWD
    ap = os.path.abspath(p)
    return ap if os.path.exists(ap) else p


# -------------------------- public API --------------------------

def fetch_events(
    session_id: Optional[str] = None,
    since_ts: Optional[float] = None,
    until_ts: Optional[float] = None,
    limit: Optional[int] = None,
    include_vectors: bool = True,
) -> List[Dict[str, Any]]:
    """
    Return raw events with handy fields normalized.
    Expected columns in `events` (based on your pipeline):
      event_id, session_id, start_ts, end_ts,
      video_path, audio_path,
      scene_type, raw, embeddings_F (BLOB)
    """
    con = connect(read_only=True)
    _ensure_aux_tables(con)

    where = []
    args: List[Any] = []
    if session_id:
        where.append("session_id = ?")
        args.append(session_id)
    if since_ts is not None:
        where.append("start_ts >= ?")
        args.append(since_ts)
    if until_ts is not None:
        where.append("end_ts <= ?")
        args.append(until_ts)

    sql = """
        SELECT event_id, session_id, start_ts, end_ts,
               video_path, audio_path, scene_type,
               raw, {veccol}
        FROM events
    """.format(veccol="embeddings_F" if include_vectors else "NULL as embeddings_F")

    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY start_ts ASC"
    if limit:
        sql += f" LIMIT {int(limit)}"

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
            "duration": (float(e["end_ts"]) - float(e["start_ts"])) if (e["start_ts"] is not None and e["end_ts"] is not None) else None,
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
    """
    Return episodes with basic metadata.
    Expected columns in `episodes`:
      episode_id, start_ts, end_ts, caption, tags_json, f_embed, summary,
      thought_text, valence_guess, confidence, salience_mean, num_events
    """
    filters = filters or {}
    con = connect(read_only=True)
    _ensure_aux_tables(con)

    where = []
    args: List[Any] = []

    # common filters
    if "session_id" in filters:
        where.append("""
            EXISTS (SELECT 1 FROM episode_events ee
                    JOIN events ev ON ev.event_id = ee.event_id
                    WHERE ee.episode_id = episodes.episode_id
                      AND ev.session_id = ?)
        """)
        args.append(filters["session_id"])

    if "min_events" in filters:
        where.append("num_events >= ?")
        args.append(int(filters["min_events"]))

    if "since_ts" in filters:
        where.append("end_ts >= ?")
        args.append(float(filters["since_ts"]))

    if "until_ts" in filters:
        where.append("start_ts <= ?")
        args.append(float(filters["until_ts"]))

    if "scene_type" in filters:
        where.append("caption LIKE ?")
        args.append(f"%{filters['scene_type']}%")

    veccol = "f_embed" if include_vectors else "NULL as f_embed"

    sql = f"""
        SELECT episode_id, start_ts, end_ts, caption, tags_json, {veccol} as f_embed,
               summary, thought_text, valence_guess, confidence,
               COALESCE(salience_mean, 0.0) as salience_mean,
               COALESCE(num_events, 0) as num_events
        FROM episodes
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    if order_by:
        sql += f" ORDER BY {order_by}"
    if limit:
        sql += f" LIMIT {int(limit)}"

    cur = con.execute(sql, args)
    rows = cur.fetchall()
    eps = _rowdicts(rows, cur.description)

    out: List[Dict[str, Any]] = []
    for ep in eps:
        out.append({
            "episode_id": ep["episode_id"],
            "start": float(ep["start_ts"]) if ep["start_ts"] is not None else None,
            "end": float(ep["end_ts"]) if ep["end_ts"] is not None else None,
            "duration": (float(ep["end_ts"]) - float(ep["start_ts"])) if (ep["start_ts"] is not None and ep["end_ts"] is not None) else None,
            "caption": ep.get("caption") or "",
            "tags": _json_or_none(ep.get("tags_json")) or [],
            "F": _maybe_vec(ep.get("f_embed")),
            "summary": ep.get("summary") or "",
            "thought_text": ep.get("thought_text") or "",
            "valence": ep.get("valence_guess"),
            "confidence": float(ep.get("confidence")) if ep.get("confidence") is not None else None,
            "salience_mean": float(ep.get("salience_mean")) if ep.get("salience_mean") is not None else 0.0,
            "num_events": int(ep.get("num_events") or 0),
        })
    return out


def iter_training_windows(
    session_id: str,
    window_sec: float = 10.0,
    stride_sec: float = 2.0,
    include_vectors: bool = True,
) -> Iterator[Dict[str, Any]]:
    """
    Yields dicts of sliding windows across a session by wall-clock time.
    Each window includes:
      - start, end, duration
      - events: [ {event_id, start, end, video_path, audio_path, scene_type, F, raw}, ... ]
      - fused_embedding_mean: mean over available event F (if any and include_vectors=True)

    Windows are aligned on start_ts; any event overlapping the window is included.
    """
    events = fetch_events(session_id=session_id, include_vectors=include_vectors)
    if not events:
        return  # nothing to yield

    # Sort just in case
    events.sort(key=lambda e: e["start"] or 0.0)
    t0 = float(events[0]["start"] or 0.0)
    tN = float(events[-1]["end"] or events[-1]["start"] or t0)

    if np is not None:
        empty_vec = None

    cur = t0
    while cur < tN:
        w_start = cur
        w_end = cur + window_sec

        # collect overlapping events
        bucket: List[Dict[str, Any]] = []
        vecs = []
        for e in events:
            es, ee = float(e["start"] or 0.0), float(e["end"] or (e["start"] or 0.0))
            if ee < w_start or es > w_end:
                continue
            bucket.append(e)
            if include_vectors and np is not None and e.get("F") is not None:
                vecs.append(e["F"])

        fused_mean = None
        if include_vectors and np is not None and len(vecs) > 0:
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
    """
    Record human/agent feedback on an episode.
    Typical labels: 'keep', 'skip', 'interesting', 'boring', etc.
    """
    con = connect(read_only=False)
    _ensure_aux_tables(con)
    with con:
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
    Upsert a record for an agent run (start/end/outcome).
    Call once at start (with outcome='started') and once at end (with final outcome).
    """
    con = connect(read_only=False)
    _ensure_aux_tables(con)
    cfg_json = json.dumps(config or {})
    det_json = json.dumps(details or {})
    with con:
        # If exists, update; else insert.
        # DuckDB supports INSERT OR REPLACE; SQLite supports UPSERT with ON CONFLICT.
        con.execute("""
            INSERT INTO agent_runs (run_id, goal, config_json, outcome, details_json, ended_ts)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
              goal=excluded.goal,
              config_json=excluded.config_json,
              outcome=excluded.outcome,
              details_json=excluded.details_json,
              ended_ts=excluded.ended_ts
        """, [run_id, goal, cfg_json, outcome, det_json, ended_ts])
