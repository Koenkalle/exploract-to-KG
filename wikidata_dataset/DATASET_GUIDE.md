# Wikidata SPARQL Query Sessions Dataset for ExplorAct

## Overview

This dataset is derived from **30 real exploratory querying sessions** on Wikidata SPARQL queries, sourced from the [ExploratoryQueryingSessions](https://github.com/hartig/ExploratoryQueryingSessions) repository.

### What is Exploratory Querying?

Exploratory querying refers to an interactive activity where users iteratively refine database queries to achieve a specific query intent. Unlike passive data exploration, it focuses on **developing the query itself**.

### Example Session

From `session01` in the original repository:

1. **Query 1**: Select all entities of type human
   - Issue: Uses wrong predicate URI for "instance of"
   - Result: Empty results

2. **Query 2**: Fix predicate URI
   - Same as Query 1 but with correct P31 predicate
   - Result: Success but slow (no LIMIT)

3. **Query 3**: Add LIMIT clause (100k)
   - User waited ~5 minutes, likely due to timeout
   - Added LIMIT to reduce results and execution time

4. **Query 4**: Increase LIMIT to 1M
   - Query 3 succeeded, so user wants more results

5. **Query 5**: Decrease LIMIT to 500k
   - User impatient again, balancing completeness vs. speed

**Query evolution pattern**: Bug fixing → Performance tuning (Categorized as "Bug Fixing" and "Result Refinement")

---

## Dataset Structure

### File Organization

```
wikidata_dataset/
├── session_repositories/
│   ├── actions.tsv          # Query transformation actions (147 actions)
│   └── displays.tsv         # Simulated query result states (177 displays)
├── raw_datasets/
│   └── 1.tsv                # Dummy graph data  representation (not used)
├── raw_query_results/
│   └── display_*.json       # Cached SPARQL query results from Wikidata
├── display_feats/
│   ├── display_pca_feats_9999.pickle    # 181-dim PCA-reduced node embeddings
│   ├── raw_display_feats.pickle         # Raw features before PCA
│   ├── raw_query_status.json            # Live vs fallback status per display
│   └── display_queries.json             # Display ID → SPARQL query mapping
├── edge/
│   ├── act_feats.pickle            # Action type one-hot (dynamic dims)
│   ├── act_five_feats.pickle       # Legacy action features (for compatibility)
│   ├── col_action.pickle           # Predicate one-hot
│   └── cond_action.pickle          # Condition one-hot
├── chunked_sessions/
│   ├── unbiased_seed_20250212.pickle    # Train/test split v1
│   └── unbiased_seed_20250214.pickle    # Train/test split v2
├── model_stats/                     # Output: model results
├── dst_probs/                       # Output: prediction probabilities
├── DATASET_GUIDE.md                 # This guide
├── FEATURE_EXTRACTION.md            # Feature extraction documentation
└── README_WIKIDATA.md               # Dataset notes
```

### Key Data Files

#### 1. `session_repositories/actions.tsv`

Records each query transformation as an action:

```
action_id | session_id | project_id | action_type | action_params | parent_display_id | child_display_id | solution
1         | 0_0       | 0          | filter      | {...}         | 1                 | 2                | 1
2         | 0_0       | 0          | filter      | {...}         | 2                 | 3                | 1
```

**Action Types** (extracted from SPARQL modifications):
- `filter`: Added/modified FILTER clauses
- `join`: Added OPTIONAL or UNION patterns
- `projection`: Changed SELECT variables
- `extension`: Added more graph traversal patterns
- `aggregation`: Added COUNT/SUM/etc. aggregate functions
- `union`: Added UNION operations

**Action Parameters** (parsed from SPARQL):
```python
{
    'field': 'http://www.wikidata.org/prop/direct/P31',  # Predicate/column
    'num_triples': 2,                                      # Graph triple patterns
    'num_filters': 1,                                      # FILTER clauses
    'limit': 500000,                                       # LIMIT value
    'offset': None,                                        # OFFSET value
    'optional': 1,                                         # OPTIONAL count
    'union': 0                                             # UNION count
}
```

#### 2. `session_repositories/displays.tsv`

Represents the simulated result state after each query:

```
display_id | session_id | query_index
1          | 0_0       | 0
2          | 0_0       | 1
3          | 0_0       | 2
```

- `display_id`: Unique identifier for this result state
- `session_id`: Which session it belongs to
- `query_index`: Position in the session sequence

#### 3. Feature Files

**Node Features** (`display_pca_feats_9999.pickle`):
- 181-dimensional embeddings per display
- Represents the result/state of a query
- **Extracted from actual SPARQL query results** via live Wikidata endpoint queries
- Features computed from result bindings (numeric stats, categorical histograms)
- PCA-reduced from variable-length raw features to fixed 181 dimensions
- See `FEATURE_EXTRACTION.md` for detailed methodology
- Format: `{display_id: np.array([...181 dims...])}` 

**Edge Features** (action metadata):
- `act_feats.pickle`: One-hot encoding for action type
  - Dimensions depend on extracted action types (filter, join, projection, etc.)
  - Format: `{action_id: np.array([...])}`
  - Legacy `act_five_feats.pickle` also provided for compatibility

- `col_action.pickle`: One-hot for predicate/column
  - Dimensions depend on predicates extracted from queries
  - Common predicates: P31 (instance of), P625 (coordinates), P27 (country), etc.
  - Format: `{action_id: np.array([...])}`

- `cond_action.pickle`: One-hot for condition/operator
  - Covers comparison operators, FILTER types
  - Format: `{action_id: np.array([...])}`

#### 4. Train/Test Splits (`chunked_sessions/`)

Cross-validation chunks for leave-one-out testing:

```python
# Format: {project_id: [[[parent, child, {'aid': action_id}], ...], ...]}
{
    0: [
        [[1, 2, {'aid': 1}], [2, 3, {'aid': 2}], ...],  # Session 0 edges
        [[...]],  # Session 1 edges
    ],
    1: [...],
    2: [...],
    3: [...]
}
```

- **Projects**: 4 projects (0-3) for leave-one-out cross-validation
- **Sessions**: Multiple sessions per project (real session data)
- **Edges**: Graph edges representing action transitions
  - Format: `[parent_display_id, child_display_id, {'aid': action_id}]`

---

## How It Maps to ExplorAct Concepts

### SPARQL Queries → Analysis Tree

In ExplorAct paper terminology:

| ExplorAct Concept | Wikidata Mapping |
|---|---|
| **Analysis Tree (Ψ)** | SPARQL query session progression |
| **Action (α)** | SPARQL query transformation (filter, join, etc.) |
| **Result (Δᵣ)** | Query result state (represented by display_id) |
| **δ-Context Tree** | Last δ queries in sequence |
| **Node Feature** | Query result embedding (181-dim, from live results) |
| **Edge Feature** | Query operation metadata (dynamic dims) |

### Prediction Tasks

Given the last δ queries (context tree), predict the next query operation:

1. **τ-rec (Action-Type Recommendation)**
   - Predict: Which type of operation? (Filter, Join, Projection, etc.)
   - Features: Action type one-hot (dynamic dims based on extracted types)

2. **a-rec (Column/Predicate Recommendation)**
   - Predict: Which predicate? (P31, P625, P27, etc.)
   - Features: Predicate one-hot (dynamic dims based on extracted predicates)

3. **(τ,a)-rec (Joint Recommendation)**
   - Predict: Which operation AND which predicate?
   - Features: Concatenated action + predicate one-hots

---

## Important Configuration

### NO Logic Error Filtering

Unlike the original REACT-IDA benchmark dataset, this dataset has no hardcoded error display IDs. The REACT-IDA dataset includes a list of ~40 display IDs that represent error states in cybersecurity logs.

**Always pass empty list when using this dataset:**

```python
# In ea_sp.py, ea_mp.py, react.py:
logic_error_displays=[]  # Disable error filtering for Wikidata dataset
```

This is already handled in the ExplorAct scripts. To verify:


---

## Running ExplorAct Models

### Quick Start

```bash
# From wikidata_dataset directory, run models using parent scripts
cd /home/kasper/Reps/exploract/wikidata_dataset

# Single-Path Model (EA-SP) - Action Type Prediction
python ../ea_sp.py act 20250212 5 0

# Multi-Path Model (EA-MP) - Column Prediction  
python ../ea_mp.py col 20250212 5 0

# Baseline - Tree Edit Distance (REACT)
python ../react.py 20250212 5 0
```

**Note**: The scripts automatically use relative paths, so running from within the `wikidata_dataset/` directory ensures they find the correct data files.

### Model Selection

| Model | Script | Task | Best For |
|---|---|---|---|
| **EA-SP** | `ea_sp.py` | act, col, tg | Single-path context encoding |
| **EA-MP** | `ea_mp.py` | act, col, tg | Multi-granularity context (slower but more accurate) |
| **REACT** | `react.py` | act, col, tg | Baseline using tree edit distance |

### Parameters

- **task**: `act` (action type), `col` (column), `tg` (joint action-column)
- **seed**: Random seed (integer, e.g., 20250212)
- **main_size**: Context size (1-8, typical 5-6)
- **test_id**: Project to hold out for testing (0-3)


### Expected Output

Models produce two output files per run:

```
model_stats/
└── {task}_{seed}_{main_size}_{test_id}_gine_seq.pickle
    # Contains: {'ra3': [...], 'mrr': [...]} for 5 runs

dst_probs/
└── gine_seq_{task}_best_ra3_{seed}_{test_id}_{main_size}.pickle
    # Contains: Prediction probabilities for analysis
```

---

## Key Statistics

| Metric | Value |
|---|---|
| **Total Sessions** | 30 (from ExploratoryQueryingSessions) |
| **Projects** | 4 (cross-validation groups) |
| **Total Actions** | 147 |
| **Total Displays** | 177 |
| **Max Display ID** | 177 |
| **Max Action ID** | 147 |
| **Queries per Session** | 3-5 (varies) |
| **Node Feature Dims** | 181 (PCA-reduced) |
| **Edge Feature Dims** | Dynamic (action types + predicates) |

---

## Differences from Original REACT-IDA

| Aspect | REACT-IDA | Wikidata |
|---|---|---|
| **Domain** | Cybersecurity logs | Wikidata SPARQL queries |
| **Operations** | Packet filtering, aggregation | Graph query transformations |
| **Error States** | Yes (40 hardcoded IDs) | No (clean data, but "wrong" queries) |
| **Display IDs** | 427-4067 | 1-177 |
| **Data Volume** | ~1000s of actions | 147 actions |
| **Sessions** | Synthetic user interaction logs | Real SPARQL query logs |
| **Node Features** | From packet data | From live SPARQL query results |

---

## Citation

**Wikidata Exploratory Querying Dataset:**

Marcelo Arenas, Enrico Franconi, Janik Hammerer, Olaf Hartig, Katja Hose, Laura Koesten, George Konstantinidis, Leonid Libkin, Wim Martens, Yuya Sasaki, Stefanie Scherzinger, Katherine Thornton, and Hsiang-Yun Wu. 2025. **Exploring Exploratory Querying.** *Proceedings of the VLDB Endowment* 18(13): 5731-5739.
[PDF](https://www.vldb.org/pvldb/vol18/p5731-konstantinidis.pdf)

**Wikidata SPARQL Logs (source):**

Stanislav Malyshev, Markus Krötzsch, Larry González, Julius Gonsior, and Adrian Bielefeldt. 2018. **Getting the Most out of Wikidata: Semantic Technology Usage in Wikipedia's Knowledge Graph.** In *Proceedings of the 17th International Semantic Web Conference (ISWC)*.
[DOI](https://doi.org/10.1007/978-3-030-00668-6_23)

---

## License

This dataset is derived from the ExploratoryQueryingSessions repository, which is licensed under [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/).
