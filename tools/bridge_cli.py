# aethermind-awareness/tools/bridge_cli.py
from __future__ import annotations
import argparse, json, sys
from datastore.bridge import (
    fetch_episodes, fetch_events, fetch_episode_members,
    top_episodes_for_stitching, record_feedback, log_agent_run,
    iter_training_windows
)

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("episodes")
    s1.add_argument("--session-id")
    s1.add_argument("--min-events", type=int, default=0)
    s1.add_argument("--limit", type=int, default=10)

    s2 = sub.add_parser("episode-members")
    s2.add_argument("--episode-id", required=True)

    s3 = sub.add_parser("events")
    s3.add_argument("--session-id", required=True)
    s3.add_argument("--limit", type=int)

    s4 = sub.add_parser("top-stitch")
    s4.add_argument("--min-events", type=int, default=3)
    s4.add_argument("--k", type=int, default=10)

    s5 = sub.add_parser("feedback")
    s5.add_argument("--episode-id", required=True)
    s5.add_argument("--label", required=True)
    s5.add_argument("--note")

    s6 = sub.add_parser("runlog")
    s6.add_argument("--run-id", required=True)
    s6.add_argument("--goal", required=True)
    s6.add_argument("--outcome", required=True)
    s6.add_argument("--config")
    s6.add_argument("--details")
    s6.add_argument("--ended-ts", type=float)

    s7 = sub.add_parser("windows")
    s7.add_argument("--session-id", required=True)
    s7.add_argument("--win", type=float, default=10.0)
    s7.add_argument("--stride", type=float, default=2.0)

    args = ap.parse_args()

    if args.cmd == "episodes":
        eps = fetch_episodes(
            filters={"session_id": args.session_id, "min_events": args.min_events} if args.session_id else {"min_events": args.min_events},
            limit=args.limit
        )
        print(json.dumps(eps, indent=2, ensure_ascii=False))
    elif args.cmd == "episode-members":
        print(json.dumps(fetch_episode_members(args.episode_id), indent=2, ensure_ascii=False))
    elif args.cmd == "events":
        print(json.dumps(fetch_events(session_id=args.session_id, limit=args.limit), indent=2, ensure_ascii=False))
    elif args.cmd == "top-stitch":
        print("\n".join(top_episodes_for_stitching(args.min_events, args.k)))
    elif args.cmd == "feedback":
        record_feedback(args.episode_id, args.label, args.note)
        print("ok")
    elif args.cmd == "runlog":
        cfg = json.loads(args.config) if args.config else None
        det = json.loads(args.details) if args.details else None
        log_agent_run(args.run_id, args.goal, args.outcome, cfg, det, args.ended_ts)
        print("ok")
    elif args.cmd == "windows":
        for w in iter_training_windows(args.session_id, args.win, args.stride):
            print(json.dumps({
                "start": w["start"], "end": w["end"],
                "n_events": len(w["events"])
            }))
    else:
        ap.print_help()
        sys.exit(2)

if __name__ == "__main__":
    main()
