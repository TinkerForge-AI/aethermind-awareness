# symbol_annotation.py (replace aggregate_symbol_tags / guess_symbol_category)

# --- Episode-symbol mapping ---
def create_episode_symbols():
    con = get_con()
    print("[DEBUG] Creating episode_symbols table and populating episode-symbol mappings...")
    con.execute("""
        CREATE TABLE IF NOT EXISTS episode_symbols (
            episode_id TEXT PRIMARY KEY,
            symbol_id TEXT,
            top_tag TEXT,
            top_tag_confidence FLOAT,
            note TEXT
        )
    """)
    # For each episode, find the symbol with the most member events from that episode
    rows = con.execute("""
        SELECT ep.episode_id, sm.symbol_id, COUNT(*) as n
        FROM episodes ep
        JOIN events e ON ep.episode_id = e.episode_id
        JOIN symbol_members sm ON e.event_id = sm.event_id
        GROUP BY ep.episode_id, sm.symbol_id
        ORDER BY ep.episode_id, n DESC
    """).fetchall()
    # Map: episode_id -> (symbol_id, count)
    episode_best_symbol = {}
    for episode_id, symbol_id, n in rows:
        if episode_id not in episode_best_symbol:
            episode_best_symbol[episode_id] = (symbol_id, n)
    # For each episode, get top tag and confidence from symbol
    for episode_id, (symbol_id, n) in episode_best_symbol.items():
        tag_row = con.execute("SELECT tags_json FROM symbols WHERE symbol_id = ?", [symbol_id]).fetchone()
        top_tag, top_conf = None, None
        if tag_row and tag_row[0]:
            dist = json.loads(tag_row[0])
            if dist:
                top_tag, top_conf = max(dist.items(), key=lambda kv: kv[1])
        con.execute("""
            INSERT OR REPLACE INTO episode_symbols (episode_id, symbol_id, top_tag, top_tag_confidence, note)
            VALUES (?, ?, ?, ?, NULL)
        """, [episode_id, symbol_id, top_tag, top_conf])
        print(f"[DEBUG] Episode {episode_id}: symbol={symbol_id}, top_tag={top_tag}, conf={top_conf}")
    print(f"[symbol_annotation] Populated episode_symbols table with {len(episode_best_symbol)} records.")

import duckdb, json, os, math
from collections import Counter, defaultdict

def _ns(tag):  # optional namespacing if you store modalities separately
    # e.g., tag already includes modality like "vision:merchant"
    return tag

def get_con():
    DB_PATH = os.environ.get("AETHERMIND_DB_PATH", "aethermind.duckdb")
    return duckdb.connect(DB_PATH)

def aggregate_symbol_tags():
    con = get_con()
    print("[DEBUG] Aggregating symbol tags from member events...")
    rows = con.execute("""
        SELECT sm.symbol_id, ep.tags
        FROM symbol_members sm
        JOIN events e ON sm.event_id = e.event_id
        JOIN episodes ep ON e.episode_id = ep.episode_id
        WHERE ep.tags IS NOT NULL
    """).fetchall()
    print(f"[DEBUG] Retrieved {len(rows)} symbol-member episode tag rows.")

    per_sym_counts = defaultdict(lambda: Counter())
    per_sym_weight  = defaultdict(lambda: Counter())
    per_sym_n = Counter()

    for symbol_id, tags in rows:
        per_sym_n[symbol_id] += 1
        # tags is expected to be an array (DuckDB returns as JSON string or Python list)
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except Exception as e:
                print(f"[WARN] Could not parse tags for symbol {symbol_id}: {tags} ({e})")
                tags = []
        print(f"[DEBUG] Symbol {symbol_id}: episode tags={tags}")
        for t in tags:
            if isinstance(t, str):
                label, score = _ns(t), 1.0
            elif isinstance(t, dict):
                label, score = _ns(t.get("label","unknown")), float(t.get("score", 1.0))
            else:
                label, score = str(t), 1.0
            per_sym_counts[symbol_id][label] += 1
            per_sym_weight[symbol_id][label]  += score

    # Compute tf (weight), df, idf
    df = Counter()
    for sym, cnts in per_sym_counts.items():
        for label in cnts.keys():
            df[label] += 1
    n_symbols = len(per_sym_counts) or 1

    report_lines = []
    for symbol_id in per_sym_counts:
        tfidf = {}
        weight_sum = sum(per_sym_weight[symbol_id].values()) or 1.0
        for label, w in per_sym_weight[symbol_id].items():
            tf = w / weight_sum
            idf = math.log((n_symbols + 1) / (df[label] + 1)) + 1.0
            tfidf[label] = tf * idf
            print(f"[DEBUG] Symbol {symbol_id}: label={label}, tf={tf:.3f}, idf={idf:.3f}, tfidf={tfidf[label]:.3f}")

        # Normalize to a distribution
        Z = sum(tfidf.values()) or 1.0
        dist = {k: v / Z for k, v in tfidf.items()}
        print(f"[DEBUG] Symbol {symbol_id}: normalized tag distribution={dist}")
        # Entropy (purity proxy)
        entropy = -sum(p * math.log(p + 1e-9) for p in dist.values())
        print(f"[DEBUG] Symbol {symbol_id}: tag entropy={entropy:.3f}")
        # Top tags
        top = sorted(dist.items(), key=lambda kv: -kv[1])[:10]
        print(f"[DEBUG] Symbol {symbol_id}: top tags={top}")

        con.execute("""
            UPDATE symbols
            SET tags_json = ?,    -- store {label: prob} distribution
                tag_entropy = ?,  -- float
                support_n = ?     -- int: member events
            WHERE symbol_id = ?
        """, [json.dumps(dist), float(entropy), int(per_sym_n[symbol_id]), symbol_id])

        # Add to report
        report_lines.append(f"Symbol {symbol_id}: entropy={entropy:.3f}, top tags={top}")

    # Write report to concept_learning_report.log
    log_path = os.path.join(os.path.dirname(__file__), "concept_learning_report.log")
    with open(log_path, "w") as f:
        f.write("[Concept Learning Report]\n")
        for line in report_lines:
            f.write(line + "\n")
        f.write("[End of Report]\n")
    print(f"[symbol_annotation] Wrote concept learning report to {log_path}")
    print(f"[symbol_annotation] Updated weighted tags for {len(per_sym_counts)} symbols.")

def guess_symbol_category():
    con = get_con()
    print("[DEBUG] Guessing symbol categories based on tag distributions...")
    rows = con.execute("SELECT symbol_id, tags_json FROM symbols WHERE tags_json IS NOT NULL").fetchall()
    print(f"[DEBUG] Retrieved {len(rows)} symbols for category guessing.")
    for symbol_id, tags_json in rows:
        dist = json.loads(tags_json) if tags_json else {}
        print(f"[DEBUG] Symbol {symbol_id}: tag distribution={dist}")
        if not dist:
            continue
        label, p = max(dist.items(), key=lambda kv: kv[1])
        print(f"[DEBUG] Symbol {symbol_id}: guessed category={label}, confidence={p:.3f}")
        con.execute("""
            UPDATE symbols
            SET category_guess = ?, category_confidence = ?
            WHERE symbol_id = ?
        """, [label, float(p), symbol_id])
    print("[symbol_annotation] Category guesses updated with confidence.")

# Run symbol annotation when executed as a script
if __name__ == "__main__":
    print("[symbol_annotation] Running symbol annotation...")
    aggregate_symbol_tags()
    guess_symbol_category()
    create_episode_symbols()
    print("[symbol_annotation] Done.")