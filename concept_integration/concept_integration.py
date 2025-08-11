"""
This module reads from the existing symbols and symbol_members 
(optionally events, episodes) and writes into concepts, concept_members, and concept_aliases.

Tune thresholds:

--merge-th 0.82 (auto-merge)
--review-th 0.74 (printed for HITL review)
--min-events 3 and/or --min-sessions 2 to require support before merging.

If symbol count is huge, we may want to add ANN/alias blocking later; 
this version handles moderate N cleanly.
"""

# concept_integration.py
import os, json, math, hashlib, sys
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

import duckdb
import numpy as np

# --------------------------- DB ---------------------------

def get_con():
    DB_PATH = os.environ.get("AETHERMIND_DB_PATH", "aethermind.duckdb")
    return duckdb.connect(DB_PATH)

# ------------------------ Utilities -----------------------

def safe_json_loads(x):
    if x is None:
        return None
    if isinstance(x, (dict, list)):
        return x
    try:
        return json.loads(x)
    except Exception:
        return None

def cosine_vec(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def cosine_from_counts(ca: Dict[str, float], cb: Dict[str, float]) -> float:
    if not ca or not cb:
        return 0.0
    # dot
    inter_keys = set(ca.keys()) & set(cb.keys())
    dot = sum(ca[k] * cb[k] for k in inter_keys)
    # norms
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(dot / (na * nb))

def jaccard_set(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def stable_concept_id(symbol_ids: List[str]) -> str:
    s = "|".join(sorted(symbol_ids)).encode("utf-8")
    h = hashlib.md5(s).hexdigest()[:12]
    return f"c_{h}"

# ----------------------- Data Model -----------------------

@dataclass
class SymRow:
    symbol_id: str
    tag_counts: Dict[str, float]         # tag -> count
    aliases: Set[str]                    # alias strings (from tags, captions, notes, episode tags)
    centroid: Optional[np.ndarray]       # embedding vector (float32) or None
    event_count: int                     # support
    session_ids: Set[str]                # sessions where this symbol appears

# -------------------- Loading Symbols ---------------------

def load_symbols(con) -> Dict[str, SymRow]:
    print("[DEBUG] Loading symbols and building support/captions/episode tags maps…")
    print(f"[DEBUG] Loaded {len(sym_rows)} symbols from DB.")
    print(f"[DEBUG] Built support map for {len(support_map)} symbols.")
    print(f"[DEBUG] Built captions map for {len(captions_map)} symbols.")
    print(f"[DEBUG] Built episode tags map for {len(ep_tags_map)} symbols.")
    print(f"[DEBUG] Composed SymRow for {len(out)} symbols.")
    # Base symbol metadata
    sym_rows = con.execute("""
        SELECT symbol_id, tags_json, centroid, note
        FROM symbols
    """).fetchall()

    # Build per-symbol event and session support if available
    support_rows = []
    try:
        support_rows = con.execute("""
            SELECT sm.symbol_id,
                   COUNT(*) AS event_count,
                   COUNT(DISTINCT e.session_id) AS sess_n,
                   LIST(DISTINCT e.session_id) AS sess_list
            FROM symbol_members sm
            LEFT JOIN events e ON e.event_id = sm.event_id
            GROUP BY sm.symbol_id
        """).fetchall()
    except Exception:
        # Fallback: just count symbol_members rows; session info unknown
        support_rows = con.execute("""
            SELECT sm.symbol_id,
                   COUNT(*) AS event_count
            FROM symbol_members sm
            GROUP BY sm.symbol_id
        """).fetchall()
        support_rows = [(r[0], r[1], 0, []) for r in support_rows]

    support_map = {}
    for r in support_rows:
        sid = r[0]
        evc = int(r[1] or 0)
        sess_list = r[3] or []
        # DuckDB LIST may come back as python list already
        sess_ids = set([str(x) for x in (sess_list or []) if x is not None])
        support_map[sid] = (evc, sess_ids)

    # Optional captions & episode tags → aliases
    captions_map = defaultdict(list)
    ep_tags_map = defaultdict(list)
    try:
        cap_rows = con.execute("""
            SELECT sm.symbol_id, e.caption_event
            FROM symbol_members sm
            JOIN events e ON e.event_id = sm.event_id
            WHERE e.caption_event IS NOT NULL
        """).fetchall()
        for sid, cap in cap_rows:
            if cap:
                captions_map[sid].append(str(cap))
    except Exception:
        pass

    try:
        ep_rows = con.execute("""
            SELECT sm.symbol_id, ep.tags
            FROM symbol_members sm
            JOIN events e ON e.event_id = sm.event_id
            JOIN episodes ep ON ep.episode_id = e.episode_id
            WHERE ep.tags IS NOT NULL
        """).fetchall()
        for sid, tags in ep_rows:
            parsed = safe_json_loads(tags)
            if isinstance(parsed, list):
                ep_tags_map[sid].extend([str(t) for t in parsed])
            elif isinstance(parsed, dict):
                ep_tags_map[sid].extend([str(k) for k in parsed.keys()])
    except Exception:
        pass

    # Compose SymRow
    out: Dict[str, SymRow] = {}
    for symbol_id, tags_json, centroid_blob, note in sym_rows:
        # tag_counts: accept list[str] or dict[str->count/score]
        tag_counts: Dict[str, float] = {}
        parsed = safe_json_loads(tags_json)
        if isinstance(parsed, list):
            for t in parsed:
                tag_counts[str(t)] = tag_counts.get(str(t), 0.0) + 1.0
        elif isinstance(parsed, dict):
            for k, v in parsed.items():
                try:
                    tag_counts[str(k)] = float(v)
                except Exception:
                    tag_counts[str(k)] = 1.0

        # aliases from tags + note + captions + episode tags
        aliases: Set[str] = set(tag_counts.keys())
        if note:
            aliases.add(str(note))
        for c in captions_map.get(symbol_id, []):
            aliases.add(c)
        for t in ep_tags_map.get(symbol_id, []):
            aliases.add(t)

        # centroid decode
        centroid = None
        if centroid_blob is not None:
            try:
                centroid = np.frombuffer(centroid_blob, dtype=np.float32)
            except Exception:
                centroid = None

        evc, sess_ids = support_map.get(symbol_id, (0, set()))

        out[symbol_id] = SymRow(
            symbol_id=symbol_id,
            tag_counts=tag_counts,
            aliases=aliases,
            centroid=centroid,
            event_count=int(evc),
            session_ids=set(sess_ids),
        )
    return out

# -------------------- Merge Scoring -----------------------

def session_jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def merge_score(A: SymRow, B: SymRow,
    print(f"[DEBUG] merge_score: {A.symbol_id} vs {B.symbol_id}")
    print(f"        alias_sim={alias_sim:.3f}, tag_sim={tag_sim:.3f}, emb_sim={emb_sim:.3f}, ctx_sim={ctx_sim:.3f}, score={score:.3f}")
                w_alias=0.30, w_tag=0.30, w_emb=0.30, w_ctx=0.10) -> Tuple[float, Dict[str,float]]:
    alias_sim = jaccard_set(A.aliases, B.aliases)
    tag_sim   = cosine_from_counts(A.tag_counts, B.tag_counts)
    emb_sim   = cosine_vec(A.centroid, B.centroid)
    ctx_sim   = session_jaccard(A.session_ids, B.session_ids)
    score = w_alias*alias_sim + w_tag*tag_sim + w_emb*emb_sim + w_ctx*ctx_sim
    return score, {
        "alias": alias_sim,
        "tags": tag_sim,
        "embed": emb_sim,
        "ctx": ctx_sim,
    }

# --------------------- Union-Find -------------------------

class UF:
    def __init__(self, items: List[str]):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}
    def find(self, x: str) -> str:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]
    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

# ------------------- Concept Building ---------------------

def aggregate_cluster(symbol_ids: List[str], S: Dict[str, SymRow],
                      max_aliases=64) -> Tuple[int, Dict[str,float], Optional[bytes], List[str]]:
    # size = total event_count (fallback to number of symbols)
    size = sum(S[sid].event_count for sid in symbol_ids)
    if size <= 0:
        size = len(symbol_ids)

    # tags rollup (sum counts)
    tag_counts: Dict[str, float] = defaultdict(float)
    for sid in symbol_ids:
        for k, v in S[sid].tag_counts.items():
            tag_counts[k] += float(v)

    # centroid = mean of available centroids
    vecs = [S[sid].centroid for sid in symbol_ids if S[sid].centroid is not None]
    centroid_blob = None
    if vecs:
        L = min(len(v) for v in vecs)
        if L > 0:
            M = np.stack([v[:L] for v in vecs], axis=0).astype(np.float32)
            mean_vec = M.mean(axis=0)
            centroid_blob = mean_vec.tobytes()

    # aliases union (capped)
    alias_union: Counter = Counter()
    for sid in symbol_ids:
        alias_union.update({a: 1 for a in S[sid].aliases})
    aliases_sorted = [a for a, _ in alias_union.most_common(max_aliases)]

    return size, dict(tag_counts), centroid_blob, aliases_sorted

# ------------------- Main Integration ---------------------

def build_concepts(merge_th=0.82, review_th=0.74,
    print(f"[DEBUG] build_concepts: merge_th={merge_th}, review_th={review_th}, min_events={min_events}, min_sessions={min_sessions}, candidate_floor={candidate_floor}, max_aliases={max_aliases}")
    print(f"[DEBUG] Symbol IDs: {ids}")
    print(f"[DEBUG] Blacklist loaded: {blacklist}")
    print(f"[DEBUG] Scoring pair: {a} vs {b}")
            print(f"[DEBUG] rough_alias={rough_alias:.3f}, rough_tag={rough_tag:.3f}, rough_emb={rough_emb:.3f}, rough_ctx={rough_ctx:.3f}")
            print(f"[DEBUG] Final score={score:.3f}, comps={comps}")
    print(f"[DEBUG] Clusters formed: {concept_clusters}")
    print(f"[DEBUG] aggregate_cluster: {symbol_ids}")
    print(f"[DEBUG] Aggregated size={size}, tag_counts={tag_counts}, aliases_sorted={aliases_sorted}")
                   min_events=2, min_sessions=1,
                   candidate_floor=0.50,
                   max_aliases=64):
    """
    - Load symbols (+support & context).
    - Compute pairwise merge scores (pruned by candidate_floor).
    - Union pairs >= merge_th, respecting min support.
    - Emit concepts, concept_members, concept_aliases.
    - Print ambiguous pairs (review zone).
    """
    con = get_con()
    print("[concept_integration] Loading symbols…")
    S = load_symbols(con)
    ids = list(S.keys())
    n = len(ids)
    print(f"[concept_integration] Loaded {n} symbol(s).")

    if n == 0:
        print("[concept_integration] No symbols found; clearing concepts tables.")
        con.execute("CREATE TABLE IF NOT EXISTS concepts (concept_id TEXT PRIMARY KEY, size INT, tags_json TEXT, centroid BLOB, aliases_json TEXT)")
        con.execute("CREATE TABLE IF NOT EXISTS concept_members (concept_id TEXT, symbol_id TEXT)")
        con.execute("CREATE TABLE IF NOT EXISTS concept_aliases (concept_id TEXT, alias TEXT)")
        con.execute("DELETE FROM concepts"); con.execute("DELETE FROM concept_members"); con.execute("DELETE FROM concept_aliases")
        return

    # Optional do-not-merge list
    blacklist = set()
    try:
        rows = con.execute("SELECT symbol_id_a, symbol_id_b FROM concept_blacklist").fetchall()
        for a, b in rows:
            if a and b:
                pair = tuple(sorted((str(a), str(b))))
                blacklist.add(pair)
        if blacklist:
            print(f"[concept_integration] Loaded blacklist pairs: {len(blacklist)}")
    except Exception:
        pass

    # Pairwise scoring (with a cheap candidate prefilter)
    # For moderate N, O(N^2) is fine. If very large, add alias blocking or ANN here.
    uf = UF(ids)
    review_pairs: List[Tuple[str, str, float, Dict[str, float]]] = []

    print("[concept_integration] Scoring pairs…")
    for i in range(n):
        for j in range(i + 1, n):
            a, b = ids[i], ids[j]
            # support constraints
            if S[a].event_count < min_events or S[b].event_count < min_events:
                continue
            if len(S[a].session_ids) < min_sessions or len(S[b].session_ids) < min_sessions:
                continue
            # blacklist
            if (a, b) in blacklist:
                continue
            # cheap candidate floor (uses any of the partial signals)
            rough_alias = jaccard_set(S[a].aliases, S[b].aliases)
            rough_tag   = cosine_from_counts(S[a].tag_counts, S[b].tag_counts)
            rough_emb   = cosine_vec(S[a].centroid, S[b].centroid)
            rough_ctx   = session_jaccard(S[a].session_ids, S[b].session_ids)
            if max(rough_alias, rough_tag, rough_emb, rough_ctx) < candidate_floor:
                continue

            score, comps = merge_score(S[a], S[b])
            if score >= merge_th:
                uf.union(a, b)
            elif score >= review_th:
                review_pairs.append((a, b, score, comps))

    # Build clusters
    clusters: Dict[str, List[str]] = defaultdict(list)
    for sid in ids:
        clusters[uf.find(sid)].append(sid)
    concept_clusters = [sorted(v) for v in clusters.values()]
    concept_clusters.sort(key=len, reverse=True)
    print(f"[concept_integration] Formed {len(concept_clusters)} concept cluster(s).")

    # Prepare tables
    con.execute("""
        CREATE TABLE IF NOT EXISTS concepts (
            concept_id TEXT PRIMARY KEY,
            size INT,
            tags_json TEXT,
            centroid BLOB,
            aliases_json TEXT
        )
    """)
    con.execute("""CREATE TABLE IF NOT EXISTS concept_members (
        concept_id TEXT, symbol_id TEXT
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS concept_aliases (
        concept_id TEXT, alias TEXT
    )""")
    con.execute("DELETE FROM concepts"); con.execute("DELETE FROM concept_members"); con.execute("DELETE FROM concept_aliases")

    # Write clusters
    for cluster in concept_clusters:
        cid = stable_concept_id(cluster)
        size, tag_counts, centroid_blob, aliases_sorted = aggregate_cluster(cluster, S, max_aliases=max_aliases)
        con.execute(
            "INSERT INTO concepts (concept_id, size, tags_json, centroid, aliases_json) VALUES (?, ?, ?, ?, ?)",
            [cid, size, json.dumps(tag_counts), centroid_blob, json.dumps(aliases_sorted)]
        )
        for sid in cluster:
            con.execute("INSERT INTO concept_members (concept_id, symbol_id) VALUES (?, ?)", [cid, sid])
        for alias in aliases_sorted:
            con.execute("INSERT INTO concept_aliases (concept_id, alias) VALUES (?, ?)", [cid, alias])

    # Print review-zone pairs (optional human-in-loop)
    if review_pairs:
        review_pairs.sort(key=lambda x: -x[2])
        print("[concept_integration] Review-zone candidate merges ({} pairs):".format(len(review_pairs)))
        for a, b, s, comps in review_pairs[:50]:
            print(f"  {a} ~ {b}  score={s:.3f}  comps={{{alias:{comps['alias']:.2f}, tags:{comps['tags']:.2f}, emb:{comps['embed']:.2f}, ctx:{comps['ctx']:.2f}}}}")
    else:
        print("[concept_integration] No review-zone pairs.")

    print("[concept_integration] Done.")

# -------------------------- CLI ---------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Concept integration via multi-view clustering + union-find")
    ap.add_argument("--merge-th", type=float, default=0.82, help="Auto-merge threshold for merge_score")
    ap.add_argument("--review-th", type=float, default=0.74, help="Review-zone threshold (prints candidates)")
    ap.add_argument("--min-events", type=int, default=2, help="Min event_count per symbol to be eligible")
    ap.add_argument("--min-sessions", type=int, default=1, help="Min distinct sessions per symbol to be eligible")
    ap.add_argument("--candidate-floor", type=float, default=0.50, help="Cheap prefilter; skip pairs with all partial sims below this")
    ap.add_argument("--max-aliases", type=int, default=64, help="Cap aliases stored per concept")
    args = ap.parse_args()

    build_concepts(
        merge_th=args.merge_th,
        review_th=args.review_th,
        min_events=args.min_events,
        min_sessions=args.min_sessions,
        candidate_floor=args.candidate_floor,
        max_aliases=args.max_aliases,
    )
