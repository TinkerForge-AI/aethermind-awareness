# aethermind-awareness
The module that gives awareness to Aethermind.

# Cognitive Ladder: Recursive Abstraction & Annotation Pattern

This project implements a recursive pattern for building increasingly meaningful representations from raw data, inspired by the Mind–Body Development Ladder (see STAGE24.md):

1. **Generate Symbol:**
    - Cluster raw events/episodes into symbols based on similarity (e.g., using embeddings).

2. **Derive Meaning:**
    - Annotate or interpret symbols, either automatically (using pre-trained models, heuristics, or metadata) or via human-in-the-loop (HITL) review.

3. **Cluster at Higher Level:**
    - Group symbols into higher-order abstractions (meta-symbols, concepts, themes).

4. **Interpret at Higher Level:**
    - Assign meaning to these abstractions, building a conceptual graph or mental model.

This cycle repeats at each stage, moving from raw data to concepts, then to reasoning and planning. Each level abstracts and interprets the previous one, recursively enriching meaning and enabling more complex cognition.

## Annotation at the Symbol Level

At the symbol level, annotation is abstracted away from direct human meaning. Instead, it can be achieved by:

- **Automatic Annotation:**
  - Use statistical properties, cluster centroids, or aggregated event-level tags to assign preliminary labels or scores to symbols.
  - Apply pre-trained models to infer possible categories or functions for each symbol.

- **Meta-Annotation:**
  - Track relationships between symbols (e.g., co-occurrence, sequence patterns) to build higher-level context.
  - Use graph-based methods to identify central or bridging symbols.

- **Human-in-the-Loop (HITL):**
  - Provide checkpoints and reports (see `learning_report.log`) for human review, enabling manual annotation or validation when needed.

Annotation at this level is less about direct semantic labels and more about building structure, context, and interpretability for future stages. As the system accumulates more data and feedback, these abstract annotations can be refined into more meaningful concepts.


## Useful SQL Queries:

###  List All Episodes/Events and Their Assigned Symbol

SELECT
    sm.event_id,
    sm.symbol_id,
    s.status,
    s.tags_json,
    s.note
FROM
    symbol_members sm
LEFT JOIN
    symbols s ON sm.symbol_id = s.symbol_id
ORDER BY
    sm.event_id ASC;

### List All Members of a Specific Symbol (Cluster)

SELECT
    sm.symbol_id,
    sm.event_id,
    sm.start_ts,
    sm.end_ts,
    e.caption,
    e.tags_json,
    e.summary
FROM
    symbol_members sm
LEFT JOIN
    events e ON sm.event_id = e.event_id
WHERE
    sm.symbol_id = 'SYM_ab12cd34'
ORDER BY
    sm.start_ts ASC;

### Find Which Symbol an Episode Belongs To

SELECT
    sm.event_id,
    sm.symbol_id,
    s.status,
    s.tags_json,
    s.note
FROM
    symbol_members sm
LEFT JOIN
    symbols s ON sm.symbol_id = s.symbol_id
WHERE
    sm.event_id = 'EVT_12345678';

### Count of Episodes per Symbol (Cluster Size)

SELECT
    sm.symbol_id,
    COUNT(sm.event_id) AS n_events,
    s.status,
    s.tags_json
FROM
    symbol_members sm
LEFT JOIN
    symbols s ON sm.symbol_id = s.symbol_id
GROUP BY
    sm.symbol_id, s.status, s.tags_json
ORDER BY
    n_events DESC;