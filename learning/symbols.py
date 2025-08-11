"""
python3 -m learning.symbols cluster --session-id session_20250808_220541
(run on 8/1/25)

Quick usage:

# Cluster everything (assign threshold 0.78 is a good start)
python3 -m learning.symbols cluster --tau-assign 0.78

# Or just a single session:
python3 -m learning.symbols cluster --session-id session_20250808_220541

# Check stats
python3 -m learning.symbols stats

# Inspect a specific cluster
python3 -m learning.symbols list --symbol-id SYM_ab12cd34

# Nuke clustering tables (careful!)
python3 -m learning.symbols reset --yes
"""

# learning/symbols.py
from __future__ import annotations
import os, json, uuid, argparse, math, sys
from typing import Any, Dict, List, Optional, Iterable, Tuple

import numpy as np
import duckdb
from dotenv import load_dotenv

# -------------------- env / globals --------------------
load_dotenv()
DB_PATH = os.environ.get("AETHERMIND_DB_PATH", "aethermind.duckdb")

# Single shared RW connection (avoids DuckDB read_only conflicts)
_CON = None
def get_con():
    global _CON
    if _CON is None:
        _CON = duckdb.connect(DB_PATH)  # RW handle
    return _CON

# -------------------- storage schema --------------------
def _ensure_tables(con) -> None:
    con.execute("""
    CREATE TABLE IF NOT EXISTS symbols (
        symbol_id    TEXT PRIMARY KEY,
        status       TEXT,               -- 'uncategorized' | 'candidate' | 'categorized'
        size_events  INTEGER,
        created_ts   TIMESTAMP DEFAULT now(),
        updated_ts   TIMESTAMP DEFAULT now(),
        centroid     BLOB,               -- float32[]
        tags_json    TEXT,
        note         TEXT
    );
    """)
    con.execute("""
    CREATE TABLE IF NOT EXISTS symbol_members (
        symbol_id  TEXT,
        event_id   TEXT,
        event_uid  TEXT,
        episode_id TEXT,
        start_ts   DOUBLE,
        end_ts     DOUBLE,
        F          BLOB,                 -- float32[]
        PRIMARY KEY (symbol_id, event_id)
    );
    """)
    # Helpful view (optional)
    con.execute("""
    CREATE VIEW IF NOT EXISTS symbol_sizes AS
    SELECT symbol_id, COUNT(*) AS n
    FROM symbol_members
    GROUP BY 1
    ORDER BY n DESC;
    """)

# -------------------- helpers --------------------
def _b2f(buf: Optional[Any]) -> Optional[np.ndarray]:
    if buf is None:
        return None
    if isinstance(buf, list):  # Handle DuckDB double[] directly
        return np.array(buf, dtype=np.float32)
    if isinstance(buf, (bytes, bytearray)):  # Handle BLOBs
        return np.frombuffer(buf, dtype=np.float32).copy()
    return None  # Return None for unsupported types

def _f2b(arr: Optional[np.ndarray]) -> Optional[bytes]:
    if arr is None: return None
    return np.asarray(arr, dtype=np.float32).tobytes()

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    da = float(np.linalg.norm(a))
    db = float(np.linalg.norm(b))
    if da == 0.0 or db == 0.0: return 0.0
    return float(np.dot(a, b) / (da * db))

def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    if n == 0: return x
    return (x / n).astype(np.float32, copy=False)

def _sym_id() -> str:
    return "SYM_" + uuid.uuid4().hex[:8]

# -------------------- IO: events / symbols --------------------
def fetch_events_for_clustering(
    session_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    print(f"[debug] Fetching events for clustering (session_id={session_id}, limit={limit})")
    con = get_con()
    _ensure_tables(con)

    where = ["embeddings_F IS NOT NULL"]
    args: List[Any] = []
    if session_id:
        where.append("session_id = ?")
        args.append(session_id)

    sql = f"""
        SELECT event_id, start_ts, end_ts, embeddings_F
        FROM events
        WHERE {' AND '.join(where)}
        ORDER BY start_ts ASC
    """
    if limit:
        sql += f" LIMIT {int(limit)}"

    print(f"[debug] Executing SQL: {sql}")
    rows = con.execute(sql, args).fetchall()
    print(f"[debug] Retrieved {len(rows)} rows")

    out: List[Dict[str, Any]] = []
    for r in rows:
        embeddings_F = _b2f(r[3]) if len(r) > 3 else None  # Ensure index is valid

        out.append({
            "event_id": r[0],
            "start_ts": float(r[1]) if r[1] is not None else None,
            "end_ts": float(r[2]) if r[2] is not None else None,
            "F": embeddings_F,
        })
    print(f"[debug] Processed {len(out)} events for clustering")
    return out

def load_symbols() -> List[Dict[str, Any]]:
    print("[debug] Loading symbols from database")
    con = get_con()
    rows = con.execute("""
        SELECT symbol_id, status, size_events, centroid, tags_json, note
        FROM symbols
    """).fetchall()
    print(f"[debug] Retrieved {len(rows)} symbols")

    syms: List[Dict[str, Any]] = []
    for r in rows:
        syms.append({
            "symbol_id": r[0],
            "status": r[1] or "uncategorized",
            "size": int(r[2] or 0),
            "centroid": _b2f(r[3]),
            "tags": json.loads(r[4]) if r[4] else [],
            "note": r[5],
        })
    print(f"[debug] Loaded {len(syms)} symbols")
    return syms

def upsert_symbol(symbol_id: str, centroid: np.ndarray, size: int,
                  status: str = "uncategorized", tags: Optional[List[str]] = None, note: Optional[str] = None) -> None:
    print(f"[debug] Upserting symbol {symbol_id} (size={size}, status={status})")
    con = get_con()
    con.execute("""
        INSERT OR REPLACE INTO symbols
          (symbol_id, status, size_events, centroid, tags_json, note, updated_ts)
        VALUES (?, ?, ?, ?, ?, ?, now())
    """, [symbol_id, status, int(size), _f2b(centroid), json.dumps(tags or []), note])

def add_member(symbol_id: str, ev: Dict[str, Any]) -> None:
    print(f"[debug] Adding event {ev['event_id']} to symbol {symbol_id}")
    con = get_con()
    # Try to fetch episode_id for this event
    episode_id = None
    try:
        row = con.execute("SELECT episode_id FROM events WHERE event_id = ?", [ev["event_id"]]).fetchone()
        if row:
            episode_id = row[0]
    except Exception as e:
        print(f"[warn] Could not fetch episode_id for event {ev['event_id']}: {e}")
    con.execute("""
        INSERT OR REPLACE INTO symbol_members
          (symbol_id, event_id, event_uid, episode_id, start_ts, end_ts, F)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, [symbol_id, ev["event_id"], ev.get("event_uid"), episode_id, ev.get("start_ts"), ev.get("end_ts"), _f2b(ev["F"])]
    )

# -------------------- clustering core --------------------
def _assign_or_create(
    symbols: List[Dict[str, Any]],
    v: np.ndarray,
    tau_assign: float,
) -> Tuple[str, bool, int, float]:
    """
    Returns (symbol_id, created_new, idx, sim)
    """
    if not symbols:
        sid = _sym_id()
        return sid, True, -1, 1.0

    # Pick best by cosine
    sims = [ _cos(v, s["centroid"]) if s["centroid"] is not None else -1.0 for s in symbols ]
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])

    if best_sim >= tau_assign:
        return symbols[best_idx]["symbol_id"], False, best_idx, best_sim
    else:
        return _sym_id(), True, -1, best_sim

def _update_centroid(old: Optional[np.ndarray], size: int, v: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Incremental mean update. Returns (new_centroid, new_size)
    """
    if old is None or size <= 0:
        return v.astype(np.float32, copy=False), 1
    new_size = size + 1
    new_c = (old * (size / new_size) + v * (1.0 / new_size)).astype(np.float32, copy=False)
    return new_c, new_size

def stream_cluster(
    tau_assign: float = 0.78,
    limit_events: Optional[int] = None,
    session_id: Optional[str] = None,
    normalize: bool = True,
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    print(f"[debug] Starting stream_cluster (tau_assign={tau_assign}, limit_events={limit_events}, session_id={session_id})")
    con = get_con()
    _ensure_tables(con)

    # Load existing symbols into memory
    symbols = load_symbols()

    # Stream events
    events = fetch_events_for_clustering(session_id=session_id, limit=limit_events)
    if verbose:
        print(f"[debug] Loaded {len(symbols)} existing symbols, {len(events)} events.")

    created = 0
    joined  = 0

    for ev in events:
        v = ev["F"]
        if v is None:  # skip if no vector
            print(f"[debug] Skipping event {ev['event_id']} (no vector)")
            continue
        if normalize:
            v = _normalize(v)

        sid, is_new, idx, sim = _assign_or_create(symbols, v, tau_assign)

        if is_new:
            # Create brand new symbol
            if verbose:
                print(f"[new] {sid}  sim={sim:.3f}  from event {ev['event_id']}")
            if not dry_run:
                upsert_symbol(sid, v, 1, status="uncategorized")
                add_member(sid, ev)
            symbols.append({
                "symbol_id": sid,
                "status": "uncategorized",
                "size": 1,
                "centroid": v,
                "tags": [],
                "note": None,
            })
            created += 1
        else:
            # Join existing symbol, update centroid + size
            s = symbols[idx]
            new_c, new_sz = _update_centroid(s["centroid"], s["size"], v)
            if normalize:
                new_c = _normalize(new_c)
            if verbose:
                print(f"[join] {s['symbol_id']}  sim={sim:.3f}  → size {s['size']}→{new_sz}  ev={ev['event_id']}")
            if not dry_run:
                upsert_symbol(s["symbol_id"], new_c, new_sz, status=s["status"], tags=s.get("tags"), note=s.get("note"))
                add_member(s["symbol_id"], ev)
            s["centroid"], s["size"] = new_c, new_sz
            joined += 1

    if verbose:
        print(f"[cluster] done. created={created}, joined={joined}, total_symbols={len(symbols)}")

# -------------------- reporting --------------------
def show_stats(top_k: int = 10) -> None:
    con = get_con()
    _ensure_tables(con)

    n_events = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    n_syms   = con.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
    n_mems   = con.execute("SELECT COUNT(*) FROM symbol_members").fetchone()[0]

    print(f"DB: {DB_PATH}")
    print(f"events: {n_events:,}")
    print(f"symbols: {n_syms:,}")
    print(f"members: {n_mems:,}")

    print("\nTop symbols by size:")
    rows = con.execute("""
        SELECT s.symbol_id, s.status, s.size_events
        FROM symbols s
        ORDER BY s.size_events DESC NULLS LAST
        LIMIT ?
    """, [top_k]).fetchall()
    for r in rows:
        sid, status, size = r
        print(f" - {sid:>12}  size={size or 0:<5}  status={status or 'uncategorized'}")

def list_symbol(symbol_id: str, limit: int = 15) -> None:
    con = get_con()
    _ensure_tables(con)

    row = con.execute("""
        SELECT symbol_id, status, size_events, tags_json, note
        FROM symbols WHERE symbol_id = ?
    """, [symbol_id]).fetchone()
    if not row:
        print(f"No such symbol: {symbol_id}")
        return
    sid, status, size, tags_json, note = row
    print(f"Symbol {sid}  status={status}  size={size}  tags={tags_json or '[]'}  note={note or ''}")

    members = con.execute("""
        SELECT event_id, event_uid, start_ts, end_ts
        FROM symbol_members
        WHERE symbol_id = ?
        ORDER BY start_ts ASC
        LIMIT ?
    """, [symbol_id, limit]).fetchall()
    for m in members:
        print(f"   • {m[0]}  {m[2]}–{m[3]}  uid={m[1]}")

def reset_symbols(confirm: bool = False) -> None:
    if not confirm:
        print("Refusing to reset without --yes.")
        return
    con = get_con()
    _ensure_tables(con)
    with con:
        con.execute("DELETE FROM symbol_members;")
        con.execute("DELETE FROM symbols;")
    print("✅ Cleared symbols and symbol_members.")

# -------------------- CLI --------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aethermind symbols: online clustering & inspection")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_cluster = sub.add_parser("cluster", help="Stream events and cluster (assign-or-create)")
    p_cluster.add_argument("--tau-assign", type=float, default=0.78, help="cosine threshold to join existing symbol")
    p_cluster.add_argument("--limit", type=int, default=None, help="limit events to process")
    p_cluster.add_argument("--session-id", type=str, default=None, help="filter to a single session")
    p_cluster.add_argument("--no-normalize", action="store_true", help="disable L2 normalization")
    p_cluster.add_argument("--dry-run", action="store_true", help="do not write to DB")
    p_cluster.add_argument("-q", "--quiet", action="store_true", help="reduce logging")

    p_stats = sub.add_parser("stats", help="Show high-level counts")
    p_stats.add_argument("--top-k", type=int, default=10)

    p_list = sub.add_parser("list", help="List a symbol and some members")
    p_list.add_argument("--symbol-id", required=True)
    p_list.add_argument("--limit", type=int, default=15)

    p_reset = sub.add_parser("reset", help="Danger: wipe symbols tables")
    p_reset.add_argument("--yes", action="store_true")

    return p

def main():
    args = build_argparser().parse_args()
    if args.cmd == "cluster":
        stream_cluster(
            tau_assign=args.tau_assign,
            limit_events=args.limit,
            session_id=args.session_id,
            normalize=not args.no_normalize,
            dry_run=args.dry_run,
            verbose=not args.quiet,
        )
        # Automated HITL report: episode-to-symbol mapping
        con = get_con()
        print("\n[HITL] Episode-to-Symbol Mapping Report:")
        rows = con.execute('''
            SELECT sm.event_id, sm.symbol_id, s.status, s.tags_json, s.note
            FROM symbol_members sm
            LEFT JOIN symbols s ON sm.symbol_id = s.symbol_id
            ORDER BY sm.event_id ASC;
        ''').fetchall()
        report_lines = []
        for r in rows:
            event_id, symbol_id, status, tags_json, note = r
            line = f"  Event {event_id} → Symbol {symbol_id} | Status: {status} | Tags: {tags_json or '[]'} | Note: {note or ''}"
            print(line)
            report_lines.append(line)
        print("[HITL] End of mapping report.\n")
        # Write report to log file in learning/
        log_path = os.path.join(os.path.dirname(__file__), "learning_report.log")
        with open(log_path, "w") as f:
            f.write("[HITL] Episode-to-Symbol Mapping Report:\n")
            for line in report_lines:
                f.write(line + "\n")
            f.write("[HITL] End of mapping report.\n")
    elif args.cmd == "stats":
        show_stats(top_k=args.top_k)
    elif args.cmd == "list":
        list_symbol(args.symbol_id, limit=args.limit)
    elif args.cmd == "reset":
        reset_symbols(confirm=args.yes)
    else:
        raise SystemExit(f"unknown cmd: {args.cmd}")

if __name__ == "__main__":
    main()
