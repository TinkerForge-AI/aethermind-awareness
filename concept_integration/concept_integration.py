# concept_integration.py
import duckdb, os, json
from collections import defaultdict

def get_con():
    DB_PATH = os.environ.get("AETHERMIND_DB_PATH", "aethermind.duckdb")
    return duckdb.connect(DB_PATH)

def build_concepts(weight_min=0.3, rel_types=("co_occurs_with","similar_to")):
    """
    Concept integration: deduplicate/merge symbols, resolve aliases, assign Concept IDs.
    Output: concepts table, alias table, and mapping from symbols/episodes/labels to Concept IDs.
    """
    con = get_con()
    print(f"[DEBUG] Fetching symbols for concept integration...")
    # Fetch all symbols and their aliases (top tags, captions)
    sym_rows = con.execute("""
        SELECT symbol_id, tags_json, centroid, note
        FROM symbols
    """).fetchall()
    print(f"[DEBUG] Retrieved {len(sym_rows)} symbols from symbols table.")

    # Alias mapping: collect all possible aliases for each symbol
    alias_map = defaultdict(set)
    symbol_tags = {}
    symbol_centroids = {}
    for symbol_id, tags_json, centroid, note in sym_rows:
        tags = json.loads(tags_json) if tags_json else {}
        symbol_tags[symbol_id] = tags
        symbol_centroids[symbol_id] = centroid
        # Add top tags as aliases
        for k in tags.keys():
            alias_map[symbol_id].add(k)
        # Add note as alias if present
        if note:
            alias_map[symbol_id].add(note)

    # Optionally, add episode-level aliases
    ep_rows = con.execute("""
        SELECT episode_id, top_tag FROM episode_symbols WHERE top_tag IS NOT NULL
    """).fetchall()
    for episode_id, top_tag in ep_rows:
        # Find symbol for this episode
        row = con.execute("SELECT symbol_id FROM episode_symbols WHERE episode_id = ?", [episode_id]).fetchone()
        if row:
            symbol_id = row[0]
            alias_map[symbol_id].add(top_tag)

    # Assign Concept IDs (one per symbol for now, can merge by alias later)
    concept_ids = {}
    for i, symbol_id in enumerate(symbol_tags.keys()):
        cid = f"c_{i:05d}"
        concept_ids[symbol_id] = cid

    # Write concepts and concept_members
    print(f"[DEBUG] Creating concepts and concept_members tables if not exist.")
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
    con.execute("DELETE FROM concepts"); con.execute("DELETE FROM concept_members")

    for symbol_id, cid in concept_ids.items():
        tags = symbol_tags[symbol_id]
        centroid = symbol_centroids[symbol_id]
        aliases = set(alias_map[symbol_id])
        size = sum(tags.values()) if tags else 0

        # Aggregate captions and episode tags from member events via symbol_members
        event_rows = con.execute("""
            SELECT e.caption_event, ep.tags
            FROM symbol_members sm
            JOIN events e ON sm.event_id = e.event_id
            JOIN episodes ep ON e.episode_id = ep.episode_id
            WHERE sm.symbol_id = ?
        """, [symbol_id]).fetchall()
        captions = []
        episode_tags = []
        for caption, ep_tags in event_rows:
            if caption:
                captions.append(caption)
            if ep_tags:
                try:
                    tags_list = json.loads(ep_tags) if isinstance(ep_tags, str) else ep_tags
                    episode_tags.extend(tags_list)
                except Exception as e:
                    print(f"[WARN] Could not parse episode tags for symbol {symbol_id}: {ep_tags} ({e})")

        # Add most common captions as aliases
        from collections import Counter
        caption_counts = Counter(captions)
        top_captions = [c for c, _ in caption_counts.most_common(3)]
        aliases.update(top_captions)

        # Add most common episode tags as aliases
        tag_counts = Counter(episode_tags)
        top_tags = [t for t, _ in tag_counts.most_common(3)]
        aliases.update(top_tags)

        con.execute("INSERT INTO concepts (concept_id, size, tags_json, centroid, aliases_json) VALUES (?, ?, ?, ?, ?)",
                    [cid, size, json.dumps(tags), centroid, json.dumps(list(aliases))])
        con.execute("INSERT INTO concept_members (concept_id, symbol_id) VALUES (?, ?)", [cid, symbol_id])
        print(f"[DEBUG] Concept {cid} (symbol {symbol_id}) written to DB with aliases: {list(aliases)}")

    # Write alias table
    print(f"[DEBUG] Creating concept_aliases table if not exist.")
    con.execute("""
        CREATE TABLE IF NOT EXISTS concept_aliases (
            concept_id TEXT,
            alias TEXT
        )
    """)
    con.execute("DELETE FROM concept_aliases")
    for symbol_id, cid in concept_ids.items():
        for alias in alias_map[symbol_id]:
            con.execute("INSERT INTO concept_aliases (concept_id, alias) VALUES (?, ?)", [cid, alias])
    print(f"[concept_integration] Built {len(concept_ids)} concept(s) and alias table.")


# Run concept integration when executed as a script
if __name__ == "__main__":
    print("[concept_integration] Running concept integration...")
    build_concepts()
    print("[concept_integration] Done.")