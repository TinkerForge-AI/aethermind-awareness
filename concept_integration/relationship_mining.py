# relationship_mining.py (add or replace functions)
import os, duckdb
import math
from collections import defaultdict, Counter
import numpy as np

def get_con():
    DB_PATH = os.environ.get("AETHERMIND_DB_PATH", "aethermind.duckdb")
    return duckdb.connect(DB_PATH)

def mine_cooccurrence(window_sec: float = 15.0, min_pmi: float = 0.0):
    """
    Co-occurrence if two symbols have member events whose time ranges overlap within window_sec.
    Uses PMI to normalize popularity. Stores results into symbol_edges.
    """
    con = get_con()
    print(f"[DEBUG] Mining co-occurrence edges (window_sec={window_sec}, min_pmi={min_pmi})")
    rows = con.execute("""
        SELECT sm.symbol_id, sm.event_id, e.session_id, e.start_ts, e.end_ts
        FROM symbol_members sm
        JOIN events e ON e.event_id = sm.event_id
        WHERE e.session_id IS NOT NULL
        ORDER BY e.session_id, e.start_ts
    """).fetchall()
    print(f"[DEBUG] Retrieved {len(rows)} symbol-member events.")

    by_session = defaultdict(list)
    for sym, ev, sess, s, e in rows:
        by_session[sess].append((sym, ev, float(s), float(e)))

    sym_count = Counter()
    pair_count = Counter()
    total_windows = 0

    for sess, items in by_session.items():
        print(f"[DEBUG] Session {sess} with {len(items)} items.")
        items.sort(key=lambda x: x[2])
        n = len(items)
        total_windows += n
        for i in range(n):
            si, ei = items[i][2], items[i][3]
            sym_i = items[i][0]
            sym_count[sym_i] += 1
            j = i + 1
            while j < n and items[j][2] - ei <= window_sec:
                sym_j = items[j][0]
                if sym_j != sym_i:
                    pair = tuple(sorted((sym_i, sym_j)))
                    pair_count[pair] += 1
                    print(f"[DEBUG] Co-occurrence: {pair} in session {sess}")
                j += 1

    N = float(total_windows) or 1.0
    edges = []
    for (a,b), c_ij in pair_count.items():
        p_i = sym_count[a] / N
        p_j = sym_count[b] / N
        p_ij = c_ij / N
        if p_i <= 0 or p_j <= 0 or p_ij <= 0:
            continue
        pmi = math.log(p_ij / (p_i * p_j))
        npmi = pmi / (-math.log(p_ij))
        if pmi >= min_pmi:
            edges.append((a, b, "co_occurs_with", float(pmi), float(npmi), c_ij))
            print(f"[DEBUG] Edge: {a} <-> {b} | PMI={pmi:.3f} | nPMI={npmi:.3f} | count={c_ij}")

    _write_edges(edges)
    print(f"[relationship_mining] wrote {len(edges)} co-occurrence edges (window={window_sec}s).")
    return edges

def mine_sequence(max_gap_sec: float = 10.0):
    """
    First-order transitions per session with maximum allowed time gap.
    """
    con = get_con()
    print(f"[DEBUG] Mining sequence edges (max_gap_sec={max_gap_sec})")
    rows = con.execute("""
        SELECT e.session_id, sm.symbol_id, sm.start_ts
        FROM symbol_members sm
        JOIN events e ON e.event_id = sm.event_id
        WHERE e.session_id IS NOT NULL
        ORDER BY e.session_id, sm.start_ts
    """).fetchall()
    print(f"[DEBUG] Retrieved {len(rows)} symbol-member events for sequence mining.")
    by_session = defaultdict(list)
    for sess, sym, ts in rows:
        by_session[sess].append((sym, float(ts)))

    trans = Counter()
    for sess, seq in by_session.items():
        print(f"[DEBUG] Session {sess} with {len(seq)} symbols.")
        for i in range(len(seq) - 1):
            s1, t1 = seq[i]
            s2, t2 = seq[i+1]
            if (t2 - t1) <= max_gap_sec and s1 != s2:
                trans[(s1, s2)] += 1
                print(f"[DEBUG] Transition: {s1} -> {s2} | gap={t2-t1:.2f}s")

    edges = [(a, b, "precedes", float(c), None, int(c)) for (a,b), c in trans.items()]
    _write_edges(edges)
    print(f"[relationship_mining] wrote {len(edges)} sequence edges (max_gap={max_gap_sec}s).")
    return edges

def mine_similarity(threshold=0.8, topk=20):
    con = get_con()
    print(f"[DEBUG] Mining similarity edges (threshold={threshold}, topk={topk})")
    rows = con.execute("SELECT symbol_id, centroid FROM symbols WHERE centroid IS NOT NULL").fetchall()
    print(f"[DEBUG] Retrieved {len(rows)} symbol centroids.")
    ids = [r[0] for r in rows]
    vecs = [np.frombuffer(r[1], dtype=np.float32) for r in rows]
    norms = [np.linalg.norm(v) + 1e-9 for v in vecs]
    vecs = [v/n for v,n in zip(vecs, norms)]

    edges = []
    for i in range(len(vecs)):
        sims = np.dot(vecs[i], np.vstack(vecs).T)
        idx = np.argsort(-sims)[:topk+1]
        for j in idx:
            if i == j: continue
            if sims[j] >= threshold:
                a, b = ids[i], ids[j]
                if a < b:
                    edges.append((a, b, "similar_to", float(sims[j]), None, None))
                    print(f"[DEBUG] Similarity: {a} <-> {b} | sim={sims[j]:.3f}")
    _write_edges(edges)
    print(f"[relationship_mining] wrote {len(edges)} similarity edges (thr={threshold}).")
    return edges

def _write_edges(edge_rows):
    """
    edge_rows: (src, dst, rel_type, weight, aux, count)
    Creates/updates `symbol_edges` table.
    """
    con = get_con()
    con.execute("""
        CREATE TABLE IF NOT EXISTS symbol_edges (
          src TEXT, dst TEXT, rel_type TEXT,
          weight DOUBLE, aux DOUBLE, count BIGINT,
          updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.executemany("""
        INSERT INTO symbol_edges (src, dst, rel_type, weight, aux, count)
        VALUES (?, ?, ?, ?, ?, ?)
    """, edge_rows)

# Run relationship mining when executed as a script
if __name__ == "__main__":
    print("[relationship_mining] Running co-occurrence mining...")
    mine_cooccurrence()
    print("[relationship_mining] Running sequence mining...")
    mine_sequence()
    print("[relationship_mining] Running similarity mining...")
    mine_similarity()
    print("[relationship_mining] Done.")