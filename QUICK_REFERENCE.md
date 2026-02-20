# ExplorAct Quick Reference - Running on New Datasets

NOTE: This file is likely outdated.

A condensed guide/reminder on how to get ExplorAct running on your own data in 5 minutes.

Uses dummy data, that in itself is meaningless.

## Key Principle

**Datasets should be self-contained and interchangeable.** Create one independently and run from within it, or use the default in the current directory.

## Approaches

### Option A: Use Default Dataset (Existing)
```bash
# From exploract root directory
python ea_sp.py act 20250212 5 0  # Uses ./session_repositories/ (default)
```

### Option B: Create and Use Dummy Dataset (Isolated)
```bash
# 1. Create self-contained dummy dataset
python create_dummy_dataset.py --output my_dataset

# 2. Run ExplorAct methods from within it
cd my_dataset
python ../ea_sp.py act 20250212 5 0    # EA-SP (action-type prediction)
python ../ea_mp.py col 20250212 5 0    # EA-MP (column prediction)
python ../react.py 20250212 5 0        # REACT baseline

# 3. Check results (isolated in my_dataset/)
ls model_stats/
```

### Option C: Create Multiple Independent Datasets
```bash
# Create different versions for testing
python create_dummy_dataset.py --output test_small --records 500
python create_dummy_dataset.py --output test_large --records 5000

# Run on each independently
cd test_small && python ../ea_sp.py act 20250212 5 0
cd ../test_large && python ../ea_sp.py act 20250212 5 0
```

## File Structure Your Dataset Needs

```
my_dataset/
├── raw_datasets/
│   └── 1.tsv                    # Your actual data (tab-separated)
├── session_repositories/
│   ├── actions.tsv              # User action logs
│   └── displays.tsv             # Display/view snapshots
├── display_feats/
│   └── display_pca_feats_*.pickle    # Node features (181-dim)
├── edge/
│   ├── act_five_feats.pickle         # Action one-hot
│   ├── col_action.pickle             # Column one-hot
│   └── cond_action.pickle            # Condition features
└── chunked_sessions/
    └── unbiased_seed_*.pickle        # Session splits for CV
```

## What Each File Contains

### `raw_datasets/1.tsv`
Your actual data. Example:
```
customer_id    amount    category    region    date
1              45.2      Electronics North     2024-01-01
2              150.5     Clothing    South     2024-01-02
3              32.1      Books       East      2024-01-03
```

**Requirements:**
- Tab-separated values
- Header row with column names
- Mix of numerical and categorical columns
- At least 100 rows

NOTE: Wikidata dataset does not use local datafiles currently. Relies on online endpoint.

### `session_repositories/actions.tsv`
User interactions. Columns:
```
action_id | action_type | action_params | session_id | user_id | project_id | creation_time | parent_display_id | child_display_id | solution
1         | filter      | {"field": "category", "term": "Electronics", "condition": 8} | 1 | 1 | 1 | 2024-01-01 10:00:00 | 1 | 2 | True
2         | group       | {"field": "region", "aggregations": []} | 1 | 1 | 1 | 2024-01-01 10:05:00 | 2 | 3 | True
```

**Key columns:**
- `action_type`: One of {filter, group, sort, projection}
- `action_params`: JSON string with operation details
- `session_id`: Groups actions into user sessions
- `project_id`: 0-indexed (0, 1, 2, 3 for leave-one-out CV)

**action_params structure by type:**
```json
// Filter action
{"field": "column_name", "term": "value", "condition": 8}

// Group action (aggregation)
{"field": "column_name", "aggregations": [{"field": "amount_col", "type": "sum"}]}

// Sort action
{"field": "column_name", "order": "asc"}

// Projection (select columns)
{"fields": ["col1", "col2"]}
```

### `session_repositories/displays.tsv`
State after each action. Example:
```
display_id | filtering | sorting | grouping | data_layer | granularity_layer | ...
1          | {...}     | {...}   | {...}    | {...}      | {...}             | ...
```

**Key columns:**
- `filtering`: JSON list of active filters
- `sorting`: JSON list of sort specs
- `grouping`: JSON list of group-by fields
- `data_layer`: JSON dict with column statistics `{"col": {"unique": 0.95, "entropy": 0.9}}`
- `granularity_layer`: JSON dict with grouping metadata `{"size_mean": 100, "group_attrs": ["category"]}`

### `display_feats/display_pca_feats_*.pickle`
Pre-computed node embeddings (one per display). Format:
```python
{
    1: np.array([0.12, 0.45, ..., 0.67]),  # 181-dimensional float32
    2: np.array([0.23, 0.34, ..., 0.78]),
    ...
}
```

**Must have:**
- 181 dimensions per node (paper uses PCA reduction)
- All float32 dtype
- Keys = display IDs from displays.tsv

### `edge/act_five_feats.pickle` (or legacy `edge/act_feats.pickle`)
One-hot encoding of action types:
```python
{
    1: np.array([1, 0, 0, 0]),  # "filter" (4 types: filter, group, sort, projection)
    2: np.array([0, 1, 0, 0]),  # "group"
    ...
}
```

- Previously the action-feature file was named `act_five_feats.pickle` (12-dim). New datasets use `act_feats.pickle` and tools support both names for compatibility.

### `edge/col_action.pickle`
One-hot encoding of columns:
```python
{
    1: np.array([1, 0, 0, 0, 0, 0, 0, 0]),  # "customer_id" (8 columns total)
    2: np.array([0, 1, 0, 0, 0, 0, 0, 0]),  # "amount"
    ...
}
```

### `edge/cond_action.pickle`
Condition/operator features:
```python
{
    1: np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),  # 10-dim uniform
    2: np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    ...
}
```

### `chunked_sessions/unbiased_seed_*.pickle`
Cross-validation splits:
```python
{
    'sessions': [[1, 0], [2, 0], [3, 0]],  # [session_id, project_id]
    'seed': 20250212,
    'projects': [0, 1, 2, 3]
}
```

## Typical Workflow

### Step 1: Prepare Your Data

Create `actions.tsv` and `displays.tsv` from your interaction logs:

```python
import pandas as pd
import json

# Assume you have user interaction history
interactions = load_your_interactions()  # [(user_id, session_id, action_type, field, ...)]

actions = []
for action_id, (user_id, session_id, action_type, field, ...) in enumerate(interactions):
    actions.append({
        'action_id': action_id,
        'action_type': action_type,  # 'filter', 'group', 'sort'
        'action_params': json.dumps({'field': field, ...}),
        'session_id': session_id,
        'user_id': user_id,
        'project_id': 0,  # Assign project (0-3 for CV)
        'creation_time': datetime.now().isoformat(),
        'parent_display_id': parent_id,
        'child_display_id': child_id,
        'solution': True  # Mark successful analyses
    })

pd.DataFrame(actions).to_csv('session_repositories/actions.tsv', sep='\t', index=False)
```

### Step 2: Generate Features

Run notebooks to extract features (or use dummy features for quick test):

```bash
# Using pre-computed features
jupyter notebook node_feat_gen.ipynb  # Generates display_pca_feats_*.pickle
jupyter notebook edge_feat_gen.ipynb  # Generates act_feats, col_feats, cond_feats

# Or use dummy features for testing
python create_dummy_dataset.py --output my_dataset
```

### Step 3: Run Models

```bash
# Navigate to dataset directory
cd my_dataset

# Test all three tasks with EA-SP (fast)
python ../ea_sp.py act 20250212 5 0   # Predict action type
python ../ea_sp.py col 20250212 5 0   # Predict column
python ../ea_sp.py tg 20250212 5 0    # Predict action+column jointly

# Use EA-MP for better accuracy (slower)
python ../ea_mp.py act 20250212 5 0
python ../ea_mp.py col 20250212 5 0

# Compare with REACT baseline
python ../react.py 20250212 5 0
```

### Step 4: Analyze Results

```python
import pickle

# Load results from one run
with open('model_stats/act_20250212_5_[0]_gine_seq.pickle', 'rb') as f:
    results = pickle.load(f)
    
print(f"Recall@3: {np.mean(results['ra3']):.2%}")  # Average over 5 runs
print(f"MRR: {np.mean(results['mrr']):.2%}")

# Load probability predictions for evidence fusion
with open('dst_probs/gine_seq_act_best_ra3_20250212_[0]_5.pickle', 'rb') as f:
    predictions = pickle.load(f)
    print(f"Predictions shape: {len(predictions)}")
```

## Command Reference

### Running Models

```bash
# EA-SP: State-Perspective (simpler, faster)
python ea_sp.py {task} {seed} {context_size} {test_project}

# EA-MP: Multi-Perspective (more accurate)
python ea_mp.py {task} {seed} {context_size} {test_project}

# REACT: Baseline (k-NN with tree edit distance)
python react.py {seed} {context_size} {test_project}
```

**Parameters:**
- `{task}`: `act` (action type) | `col` (column) | `tg` (joint)
- `{seed}`: Random seed (e.g., 20250212)
- `{context_size}`: δ ∈ {3,4,5,6,7,8} (number of recent actions)
- `{test_project}`: Project ID to hold out for testing (0-3)

### Timing Analysis

```bash
# Quick inference timing (no training)
python ea_sp_time.py {task} {context_size}

# Scaling analysis: timing vs log size
python ea_sp_time_logsize.py {seed} {context_size}

# Same for EA-MP
python ea_mp_time.py {task} {context_size}
python ea_mp_time_logsize.py {seed} {context_size}
```

### Results Analysis

```bash
# See what results are available
ls model_stats/     # Accuracy metrics
ls dst_probs/       # Probability predictions
ls time_plots/      # Timing plots (if generated)

# Load and analyze (see result_analyser.ipynb)
jupyter notebook result_analyser.ipynb
jupyter notebook evidence_fusion.ipynb
```

## Common Mistakes & Fixes

| Problem | Solution |
|---------|----------|
| "No such file: display_feats/display_pca_feats_9999.pickle" | Run `create_dummy_dataset.py` or generate features via notebooks |
| "shape mismatch" in GINE | Verify display features have 181 dimensions |
| Empty graphs / no edges | Ensure actions have valid parent_display_id → child_display_id chains |
| Bad results | Check that project_id is in {0,1,2,3}; at least 10+ actions per session |
| CUDA OOM | Use CPU: edit line 45 in ea_sp.py to `device = torch.device('cpu')` |

## Paper Parameters Used

- **Context size δ**: 3, 4, 5, 6, 7, 8 (try δ=5 first)
- **Feature dimensions**: Node=181, Edge=20
- **Batch size**: Dynamically adjusted (avoid size-1 batches)
- **Optimizer**: Adam, learning rate 0.001
- **Loss**: Binary cross-entropy (BCE)
- **Train/Test**: Leave-one-project-out (4 projects, 3-way split)
- **Runs per config**: 5 different random seeds

## Further Reading

- **Architecture**: See `.github/copilot-instructions.md` § "Three Main Variants"
- **Feature Engineering**: See `.github/copilot-instructions.md` § "Node and Edge Feature Engineering"
- **Model Details**: See paper sections 3.2-3.5 (in `2025_cikm.txt`)
- **Full Setup**: See `RUNNING_ON_NEW_DATASET.md`
