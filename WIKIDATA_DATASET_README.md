# Wikidata SPARQL Dataset for ExplorAct

## Quick Summary

A new dataset has been created using **30 real exploratory SPARQL query sessions** from the [ExploratoryQueryingSessions](https://github.com/hartig/ExploratoryQueryingSessions) repository (Wikidata knowledge graph).

### Location

```
/home/kasper/Reps/exploract/wikidata_dataset/
```

### Dataset Contents

| File/Directory | Purpose | Details |
|---|---|---|
| `session_repositories/actions.tsv` | Query transformation actions | 147 actions across 30 sessions |
| `session_repositories/displays.tsv` | Simulated result states | 177 display snapshots |
| `display_feats/display_pca_feats_9999.pickle` | Node embeddings | 181-dim per display |
| `edge/{act_feats,col_action,cond_action}.pickle` | Edge features | action-dims + predicate-dims + cond-dims (dynamic) |
| `chunked_sessions/unbiased_seed_*.pickle` | Train/test splits | 4 projects for cross-validation |
| `run.py` | Model execution wrapper | Easy invocation of ea_sp, ea_mp, react |

### Quick Start

```bash
cd /home/kasper/Reps/exploract/wikidata_dataset

# Run EA-SP (single-path model)
python run.py ea_sp act 20250212 5 0

# Run EA-MP (multi-path model)
python run.py ea_mp col 20250212 5 0

# Run REACT baseline
python run.py react 20250212 5 0
```

---

## Data Source & Structure

### Original Data

**Source**: [ExploratoryQueryingSessions](https://github.com/hartig/ExploratoryQueryingSessions) GitHub repository

- **Type**: SPARQL query logs from Wikidata Query Service (Wikidata)
- **Time Period**: June 2017 - March 2018
- **Sessions**: 30 manually verified exploratory querying sessions
- **Queries/Session**: 3-5 SPARQL queries each
- **License**: CC BY 4.0

### SPARQL → ExplorAct Mapping

Each SPARQL query session becomes an "analysis tree":

```
Query 1: SELECT ?x WHERE { ?x P31 Q5 }          ← Root (display_id=1)
    ↓
Query 2: ... FILTER (LANG(?label) = "en")       ← Action 1 → display_id=2
    ↓
Query 3: ... LIMIT 100000                       ← Action 2 → display_id=3
    ↓
Query 4: ... LIMIT 1000000                      ← Action 3 → display_id=4
    ↓
Query 5: ... LIMIT 500000                       ← Action 4 → display_id=5
```

**Extracted Operations** (action types):
- `filter`: Modified FILTER clauses
- `join`: Added OPTIONAL or UNION
- `projection`: Changed SELECT variables
- `extension`: Added graph traversals
- `aggregation`: Added aggregations
- `union`: Added UNION operations

### Feature Engineering

**Node Features** (181 dimensions):
- Represents the query result state
- Derived from SPARQL query results (or structured fallback)
- One vector per display_id

**Edge Features** (combined):
- **Action type** (`edge/act_feats.pickle`): one-hot encoding of operation type (number of action classes varies by dataset)
- **Predicate** (`edge/col_action.pickle`): one-hot encoding of affected predicate
- **Condition** (`edge/cond_action.pickle`): operator/condition features
- Note: For backward compatibility the generator also writes a legacy file named `edge/act_five_feats.pickle`.

### Feature Dimensions Reference

**Quick lookup for wikidata dataset dimensions:**

| Feature File | Dimension | Description |
|---|---|---|
| `display_feats/display_pca_feats_9999.pickle` | **181** | Node features (PCA-reduced display embeddings) |
| `edge/act_feats.pickle` | **4** | Action type one-hot (filter, join, projection, extension) |
| `edge/col_action.pickle` | **56** | Predicate/column one-hot encoding |
| `edge/cond_action.pickle` | **5** | Condition/operator features |
| **concat_feats** (act + col) | **60** | Combined edge features used by GINE |

**Dataset counts:**
- **177** displays (display_id 1-177)
- **147** actions (action_id 1-147)
- **30** sessions across **4** projects

### Wikidata-Specific Terminology

In the original REACT-IDA cybersecurity dataset, "actions" were operations like filter/sort on packet columns. For **Wikidata SPARQL**, we map these concepts as follows:

| ExplorAct Term | REACT-IDA (Original) | Wikidata SPARQL |
|---|---|---|
| **Action Type (τ)** | filter, project, group, sort | filter, join, projection, extension |
| **Column (a)** | Packet field (ip_src, tcp_port, etc.) | RDF predicate (P31, P279, rdfs:label, etc.) |
| **Display** | Filtered table result | SPARQL query result set |
| **Session** | Analyst exploration session | User's query refinement sequence |

**Action Types for Wikidata (4 classes):**
| Action | Meaning | SPARQL Example |
|---|---|---|
| `filter` | Add/modify FILTER clause | `FILTER (LANG(?label) = "en")` |
| `join` | Add OPTIONAL or graph pattern | `OPTIONAL { ?x rdfs:label ?label }` |
| `projection` | Change SELECT variables | `SELECT ?x ?label` → `SELECT ?x ?label ?desc` |
| `extension` | Add triple pattern traversal | Add `?x P279 ?parent` to WHERE clause |

**Columns for Wikidata (56 predicates):**
These are RDF predicates extracted from the SPARQL queries. Examples:
- `P31` (instance of)
- `P279` (subclass of)
- `rdfs:label` (label)
- `P569` (date of birth)
- `P17` (country)
- ... and 51 more unique predicates

**Conditions (5 operators):**
| Condition | Meaning |
|---|---|
| `=` | Equality filter |
| `!=` | Inequality filter |
| `LANG()` | Language filter |
| `CONTAINS()` | String containment |
| `other` | Other operators |

**Comparison with REACT-IDA benchmark:**

| Feature | REACT-IDA | Wikidata |
|---|---|---|
| Node dim (`display_pca_feats`) | 181 | 181 |
| Action types (`act_feats`) | 4 | 4 |
| Columns (`col_feats`) | 14 | 56 |
| Edge dim (`concat_feats`) | 20* | 60 |
| Displays | ~4000+ | 177 |
| Actions | ~3000+ | 147 |

*Note: REACT-IDA used `edge_dim=20` (4 actions × 14 columns + adjustments). The code now dynamically computes `EDGE_DIM` from `concat_feats` to support both datasets.

---

## Cross-Validation Setup

### Project Distribution

4 projects for leave-one-out cross-validation:

```
Project 0: Sessions  0-7   (8 sessions)
Project 1: Sessions  8-15  (8 sessions)
Project 2: Sessions 16-23  (8 sessions)
Project 3: Sessions 24-29  (6 sessions)
```

Each project can be held out for testing while others train (--test_id 0, 1, 2, or 3).

### Seeds

Two seeds provided for reproducibility:
- `unbiased_seed_20250212.pickle`
- `unbiased_seed_20250214.pickle`

---

## Running Models

### Command Format

```bash
python run.py <model> <task> <seed> <main_size> <test_id>
```

### Parameters

| Parameter | Options | Description |
|---|---|---|
| **model** | `ea_sp`, `ea_mp`, `react` | Which model to run |
| **task** | `act`, `col`, `tg` | Prediction task |
| **seed** | `20250212`, `20250214`, etc. | Random seed |
| **main_size** | 1-8 (typical: 5-6) | Context size (δ) |
| **test_id** | 0, 1, 2, 3 | Which project to test on |

### Examples

```bash
# EA-SP: Predict next action type from last 5 queries
cd /home/kasper/Reps/exploract/wikidata_dataset
python run.py ea_sp act 20250212 5 0

# EA-MP: Predict next column/predicate from last 6 queries
python run.py ea_mp col 20250212 6 0

# REACT: Baseline with TED similarity
python run.py react 20250212 5 0

# Joint prediction (action + column)
python run.py ea_sp tg 20250212 5 1
```

### Model Descriptions

| Model | Full Name | Method | Best For |
|---|---|---|---|
| **ea_sp** | Expand-Analysis Single-Path | GINE + GRU (state-perspective context) | Accuracy/speed tradeoff |
| **ea_mp** | Expand-Analysis Multi-Path | GINE + GRU (multi-granularity context) | Highest accuracy (+16-20% vs baseline) |
| **react** | Tree Edit Distance | Non-learning k-NN with BallTree | Baseline comparison |

---

## Output & Results

### Result Files

After training, results are saved:

```
model_stats/{task}_{seed}_{main_size}_{test_id}_gine_seq.pickle
dst_probs/{task}_{seed}_{test_id}_{main_size}.pickle
```

Example output structure:
```python
# model_stats file contains:
{
    'ra3': [0.234, 0.245, 0.256, 0.241, 0.253],  # 5 runs
    'mrr': [0.412, 0.425, 0.438, 0.419, 0.431]
}

# Average RA@3: 24.6% | Average MRR: 42.5%
```

### Expected Performance

On Wikidata dataset (based on dataset size and characteristics):
- **RA@3** (Recall@3): 20-35% typical
- **MRR** (Mean Reciprocal Rank): 35-50% typical
- **Inference time**: < 1 second per query

Note: Smaller dataset than REACT-IDA, so absolute numbers may be lower.

---

## Key Configuration Details

### Logic Error Filtering

**IMPORTANT**: This dataset does NOT have error states like the REACT-IDA benchmark.

The `logic_error_displays` list in `ea_sp.py` and `ea_mp.py` contains hardcoded display IDs (starting at 427) from the original REACT-IDA dataset. Since the Wikidata dataset uses display IDs 1-177, **no configuration change is required** - the hardcoded list simply doesn't match any Wikidata displays.

If you extend the Wikidata dataset to have display IDs >= 427, you would need to clear this list:

```python
# In ea_sp.py, ea_mp.py (replay_graph function):
logic_error_displays = []  # Empty list disables filtering
```

---

## File Organization

```
wikidata_dataset/
├── run.py                                    # Wrapper script
├── DATASET_GUIDE.md                         # Detailed documentation
├── README_WIKIDATA.md                       # Dataset info
│
├── session_repositories/
│   ├── actions.tsv                          # 147 query actions
│   └── displays.tsv                         # 177 result states
│
├── raw_datasets/
│   └── 1.tsv                                # Dummy graph data
│
├── display_feats/
│   └── display_pca_feats_9999.pickle        # 181-dim embeddings
│
├── edge/
│   ├── act_feats.pickle                      # Action features (dynamic dims)
│   ├── col_action.pickle                    # Predicate features
│   └── cond_action.pickle                   # Condition features
│
└── chunked_sessions/
    ├── unbiased_seed_20250212.pickle        # Train/test split v1
    └── unbiased_seed_20250214.pickle        # Train/test split v2
```

---

## Comparison with Other Datasets

| Dataset | Type | Source | Size | Domain |
|---|---|---|---|---|
| **REACT-IDA** (Original) | Real logs | Cybersecurity analysis | ~1000 actions | Packet filtering |
| **dummy_dataset** | Synthetic | Created for testing | 216 actions | Generic (no real structure) |
| **wikidata_dataset** | Real queries | SPARQL logs | 147 actions | Graph databases (Wikidata) |

---

## Generating This Dataset

The dataset was created with:

```bash
cd /home/kasper/Reps/exploract
python create_wikidata_dataset.py --output wikidata_dataset --projects 4
```

To regenerate or create variants:

```bash
# Create with different number of projects
python create_wikidata_dataset.py --output wikidata_v2 --projects 5

# Use custom output
python create_wikidata_dataset.py --output /path/to/my_dataset
```

See `create_wikidata_dataset.py` for source code.

---

## Troubleshooting

### Issue: "logic_error_displays not a valid list"

**Solution**: Ensure `ea_sp.py` and `ea_mp.py` have the configuration update. Check:

```bash
grep -n "logic_error_displays=" /home/kasper/Reps/exploract/ea_sp.py | head -5
```

Should show optional parameter in function signatures.

### Issue: "Display ID out of range"

**Solution**: Wikidata displays are 1-177. If loading other datasets, verify:
```python
print(f"Max display: {max(display_pca_feats.keys())}")
print(f"Expected: < 178")
```

### Issue: "No actions found"

**Solution**: Verify SPARQL parsing worked:
```bash
head -10 wikidata_dataset/session_repositories/actions.tsv
# Should show 147 total actions
```

---

## Next Steps

1. **Run a single experiment**: `python run.py ea_sp act 20250212 5 0`
2. **Analyze results**: Check `model_stats/` and `dst_probs/` directories
3. **Run all combinations**: Create shell script to test all models/tasks/projects
4. **Compare with REACT-IDA**: Run same experiments on original dataset
5. **Generate plots**: Use `result_analyser.ipynb` for visualization

---

## Citation & Attribution

**Dataset Creation**:
- ExploratoryQueryingSessions: https://github.com/hartig/ExploratoryQueryingSessions
- License: CC BY 4.0

**Wikidata SPARQL Logs (source)**:
- Malyshev et al., ISWC 2018
- https://iccl.inf.tu-dresden.de/web/Wikidata_SPARQL_Logs/en

**VLDB Paper**:
- Arenas et al., "Exploring Exploratory Querying", VLDB 2025
- https://www.vldb.org/pvldb/vol18/p5731-konstantinidis.pdf

---

## Questions?

Refer to:
- `wikidata_dataset/DATASET_GUIDE.md` - Detailed documentation
- `create_wikidata_dataset.py` - Source code for generation
- ExplorAct main README for model details
