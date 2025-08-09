from __future__ import annotations
import os, sys, json
from typing import Iterable, Dict, Any
import duckdb
import numpy as np
from dotenv import load_dotenv

load_dotenv()
DB_PATH = os.environ.get("AETHERMIND_DB_PATH", "aethermind.duckdb")

def connect(ro=False):
    return duckdb.connect(DB_PATH, read_only=ro)

def _ensure_tables(con):
    con.execute("""
    CREATE TABLE IF NOT EXISTS seeds (
        event_uid      TEXT PRIMARY KEY,
        session_id     TEXT,
        schema_major   INTEGER,
        schema_minor   INTEGER,
        created_at     TIMESTAMP,
        start_ts       DOUBLE,
        end_ts         DOUBLE,
        source         TEXT,
        video_path     TEXT,
        audio_path     TEXT,
        actions_json   TEXT,
        sync_json      TEXT,
        video_dyn_json TEXT,
        audio_dyn_json TEXT,
        system_json    TEXT,
        action_window_json TEXT,
        decision_trace_json TEXT
    );
    """)
    con.execute("""
    CREATE TABLE IF NOT EXISTS seed_event_map (
        event_uid TEXT PRIMARY KEY,
        event_id  TEXT
    );
    """)

def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            yield json.loads(line)

def ingest(path: str):
    con = connect(False)
    _ensure_tables(con)
    rows = []
    for obj in _iter_jsonl(path):
        rows.append((
            obj.get("event_uid"),
            obj.get("session_id"),
            (obj.get("schema_version") or {}).get("major"),
            (obj.get("schema_version") or {}).get("minor"),
            obj.get("created_at"),
            float(obj.get("start")) if obj.get("start") is not None else None,
            float(obj.get("end")) if obj.get("end") is not None else None,
            obj.get("source"),
            obj.get("video_path"),
            obj.get("audio_path"),
            json.dumps(obj.get("actions", []), ensure_ascii=False),
            json.dumps(obj.get("sync", {}), ensure_ascii=False),
            json.dumps(obj.get("video_dyn", {}), ensure_ascii=False),
            json.dumps(obj.get("audio_dyn", {}), ensure_ascii=False),
            json.dumps(obj.get("system", {}), ensure_ascii=False),
            json.dumps(obj.get("action_window", {}), ensure_ascii=False),
            json.dumps(obj.get("decision_trace", {}), ensure_ascii=False),
        ))
    with con:
        con.executemany("""
            INSERT OR REPLACE INTO seeds (
                event_uid, session_id, schema_major, schema_minor, created_at,
                start_ts, end_ts, source, video_path, audio_path,
                actions_json, sync_json, video_dyn_json, audio_dyn_json,
                system_json, action_window_json, decision_trace_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)

def join_to_events():
    con = connect(False)
    _ensure_tables(con)
    with con:
        con.execute("""
            INSERT OR REPLACE INTO seed_event_map (event_uid, event_id)
            SELECT s.event_uid, e.event_id
            FROM seeds s
            LEFT JOIN events e ON e.event_id = s.event_uid
        """)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("ingest")
    s1.add_argument("--path", required=True)

    s2 = sub.add_parser("join")

    s3 = sub.add_parser("info")

    args = ap.parse_args()
    if args.cmd == "ingest":
        ingest(args.path)
        print("ok: seeds ingested")
    elif args.cmd == "join":
        join_to_events()
        n = connect(True).execute("SELECT COUNT(*) FROM seed_event_map WHERE event_id IS NOT NULL").fetchone()[0]
        print(f"ok: joined {n} seeds to events")
    elif args.cmd == "info":
        con = connect(True)
        n = con.execute("SELECT COUNT(*) FROM seeds").fetchone()[0]
        m = con.execute("SELECT COUNT(*) FROM seed_event_map WHERE event_id IS NOT NULL").fetchone()[0]
        print(f"seeds: {n}, mapped_to_events: {m}")
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
