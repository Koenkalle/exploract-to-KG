# Running ExplorAct on a New Dataset
Note: This file is likely not up to date! 

This guide shows how to prepare and run ExplorAct's methods (EA-SP, EA-MP, REACT) on a new dataset. All datasets are **self-contained and interchangeable**. You can create multiple datasets independently without affecting existing data.

## Key Principle: Self-Contained Datasets

- Each dataset lives in its own directory with complete structure
- No overwrites of existing data
- Can create multiple datasets for testing/comparison
- Select which dataset to use by changing working directory
- Results are isolated within each dataset's `model_stats/` directory

## Dataset Requirements

ExplorAct requires three main components:

1. **Raw Dataset Files** (`raw_datasets/`): TSV files with actual data records
   - One file per dataset (e.g., `1.tsv`, `2.tsv`, etc.)
   - Tab-separated values with column names as headers
   - Should include relevant numerical and categorical columns

2. **Session Actions** (`session_repositories/actions.tsv`): Log of user interactions
   - Columns: `action_id`, `action_type`, `action_params`, `session_id`, `user_id`, `project_id`, `creation_time`, `parent_display_id`, `child_display_id`, `solution`
   - Maps which actions users performed and their outcomes

3. **Session Displays** (`session_repositories/displays.tsv`): Snapshot of data state after each action
   - Columns: `display_id`, `filtering`, `sorting`, `grouping`, `aggregations`, `data_layer`, `granularity_layer`, `projected_fields`, `session_id`, `user_id`, `project_id`, `solution`
   - Contains metadata about the data state (statistics, column properties)

## Step 1: Create a Dummy Dataset

### Create Sample Raw Data

```bash
mkdir -p dummy_dataset/raw_datasets
mkdir -p dummy_dataset/session_repositories
cd dummy_dataset
```

### Generate Sample Data Files

Create `raw_datasets/1.tsv` with e-commerce transaction data:

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create dummy dataset
np.random.seed(42)
n_records = 1000

dates = [datetime(2024, 1, 1) + timedelta(days=int(d)) 
         for d in np.random.exponential(scale=30, size=n_records)]
         
df = pd.DataFrame({
    'transaction_id': range(1, n_records + 1),
    'customer_id': np.random.randint(1, 100, n_records),
    'amount': np.random.exponential(scale=50, size=n_records),
    'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n_records),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n_records),
    'date': dates,
    'is_returned': np.random.choice([True, False], n_records, p=[0.1, 0.9]),
    'payment_method': np.random.choice(['Credit', 'Debit', 'PayPal'], n_records)
})

# Save as TSV (index=False is important - ExplorAct will set proper indices)
df.to_csv('raw_datasets/1.tsv', sep='\t', index=False)
print("Created raw_datasets/1.tsv with shape:", df.shape)
```

### Generate Session Actions File

Create `session_repositories/actions.tsv`:

```python
import pandas as pd
import json

actions_data = []
action_id = 1

# Simulate 3 user sessions
for session_id in range(1, 4):
    for user_id in range(1, 3):
        project_id = 1
        
        # Session 1: Filter by category
        if session_id == 1:
            actions_data.append({
                'action_id': action_id,
                'action_type': 'filter',
                'action_params': json.dumps({"field": "category", "term": "Electronics", "condition": 8}),
                'session_id': session_id,
                'user_id': user_id,
                'project_id': project_id,
                'creation_time': '2024-01-01 10:00:00',
                'parent_display_id': 1,
                'child_display_id': 2,
                'solution': True
            })
            action_id += 1
            
            # Then group by region
            actions_data.append({
                'action_id': action_id,
                'action_type': 'group',
                'action_params': json.dumps({"field": "region", "aggregations": []}),
                'session_id': session_id,
                'user_id': user_id,
                'project_id': project_id,
                'creation_time': '2024-01-01 10:05:00',
                'parent_display_id': 2,
                'child_display_id': 3,
                'solution': True
            })
            action_id += 1
            
            # Then sort by amount
            actions_data.append({
                'action_id': action_id,
                'action_type': 'sort',
                'action_params': json.dumps({"field": "amount"}),
                'session_id': session_id,
                'user_id': user_id,
                'project_id': project_id,
                'creation_time': '2024-01-01 10:10:00',
                'parent_display_id': 3,
                'child_display_id': 4,
                'solution': True
            })
            action_id += 1
        
        # Session 2: Filter by region
        elif session_id == 2:
            actions_data.append({
                'action_id': action_id,
                'action_type': 'filter',
                'action_params': json.dumps({"field": "region", "term": "North", "condition": 8}),
                'session_id': session_id,
                'user_id': user_id,
                'project_id': project_id,
                'creation_time': '2024-01-02 10:00:00',
                'parent_display_id': 5,
                'child_display_id': 6,
                'solution': True
            })
            action_id += 1
            
            # Group by category
            actions_data.append({
                'action_id': action_id,
                'action_type': 'group',
                'action_params': json.dumps({"field": "category", "aggregations": [{"field": "amount", "type": "sum"}]}),
                'session_id': session_id,
                'user_id': user_id,
                'project_id': project_id,
                'creation_time': '2024-01-02 10:05:00',
                'parent_display_id': 6,
                'child_display_id': 7,
                'solution': True
            })
            action_id += 1
        
        # Session 3: Filter by payment method, then by amount
        else:
            actions_data.append({
                'action_id': action_id,
                'action_type': 'filter',
                'action_params': json.dumps({"field": "payment_method", "term": "Credit", "condition": 8}),
                'session_id': session_id,
                'user_id': user_id,
                'project_id': project_id,
                'creation_time': '2024-01-03 10:00:00',
                'parent_display_id': 8,
                'child_display_id': 9,
                'solution': True
            })
            action_id += 1
            
            actions_data.append({
                'action_id': action_id,
                'action_type': 'filter',
                'action_params': json.dumps({"field": "amount", "term": "100", "condition": 32}),  # > 100
                'session_id': session_id,
                'user_id': user_id,
                'project_id': project_id,
                'creation_time': '2024-01-03 10:05:00',
                'parent_display_id': 9,
                'child_display_id': 10,
                'solution': True
            })
            action_id += 1

actions_df = pd.DataFrame(actions_data)
actions_df.to_csv('session_repositories/actions.tsv', sep='\t', index=False)
print("Created session_repositories/actions.tsv with shape:", actions_df.shape)
```

### Generate Session Displays File

Create `session_repositories/displays.tsv`:

```python
import pandas as pd
import json

displays_data = []

# Create display records for each action endpoint
display_id = 1
for session_id in range(1, 4):
    for user_id in range(1, 3):
        project_id = 1
        
        # Initial display (unfiltered)
        displays_data.append({
            'display_id': display_id,
            'filtering': json.dumps({"list": []}),
            'sorting': json.dumps({"list": []}),
            'grouping': json.dumps({"list": []}),
            'aggregations': None,
            'data_layer': json.dumps({
                'transaction_id': {'unique': 1.0, 'entropy': 1.0},
                'customer_id': {'unique': 0.1, 'entropy': 0.5},
                'amount': {'unique': 0.95, 'entropy': 0.9},
                'category': {'unique': 0.004, 'entropy': 0.15},
                'region': {'unique': 0.004, 'entropy': 0.15},
                'is_returned': {'unique': 0.002, 'entropy': 0.1},
                'payment_method': {'unique': 0.003, 'entropy': 0.12}
            }),
            'granularity_layer': None,
            'projected_fields': json.dumps({"list": [
                {"field": "transaction_id"},
                {"field": "customer_id"},
                {"field": "amount"},
                {"field": "category"},
                {"field": "region"},
                {"field": "is_returned"},
                {"field": "payment_method"},
                {"field": "date"}
            ]}),
            'session_id': session_id,
            'user_id': user_id,
            'project_id': project_id,
            'solution': True
        })
        display_id += 1
        
        # After each action, create a new display record
        for action_num in range(2 if session_id < 3 else 3):
            displays_data.append({
                'display_id': display_id,
                'filtering': json.dumps({"list": []}),
                'sorting': json.dumps({"list": []}),
                'grouping': json.dumps({"list": [{"field": "category", "groupPriority": 0}]}),
                'aggregations': json.dumps({"list": []}) if action_num == 1 else None,
                'data_layer': json.dumps({
                    'transaction_id': {'unique': 1.0, 'entropy': 1.0},
                    'customer_id': {'unique': 0.1, 'entropy': 0.5},
                    'amount': {'unique': 0.95, 'entropy': 0.9},
                    'category': {'unique': 0.004, 'entropy': 0.15},
                    'region': {'unique': 0.004, 'entropy': 0.15},
                    'is_returned': {'unique': 0.002, 'entropy': 0.1},
                    'payment_method': {'unique': 0.003, 'entropy': 0.12}
                }),
                'granularity_layer': json.dumps({'size_mean': 100, 'group_attrs': ['category']}) if action_num > 0 else None,
                'projected_fields': json.dumps({"list": [
                    {"field": "transaction_id"},
                    {"field": "customer_id"},
                    {"field": "amount"},
                    {"field": "category"},
                    {"field": "region"},
                    {"field": "is_returned"},
                    {"field": "payment_method"}
                ]}),
                'session_id': session_id,
                'user_id': user_id,
                'project_id': project_id,
                'solution': True
            })
            display_id += 1

displays_df = pd.DataFrame(displays_data)
displays_df.to_csv('session_repositories/displays.tsv', sep='\t', index=False)
print("Created session_repositories/displays.tsv with shape:", displays_df.shape)
```

## Step 2: Prepare Feature Vectors

Before running ExplorAct, you need to generate feature vectors for nodes and edges. This is done via PCA dimensionality reduction (see `node_feat_gen.ipynb`). For a quick test, you can create dummy features:

```python
import pickle
import numpy as np

# Create dummy node features (181-dimensional PCA-reduced)
display_pca_feats = {}
for i in range(1, 11):  # for each display
    display_pca_feats[i] = np.random.randn(181).astype(np.float32)

with open('display_feats/display_pca_feats_9999.pickle', 'wb') as f:
    pickle.dump(display_pca_feats, f, protocol=pickle.HIGHEST_PROTOCOL)

# Create dummy edge features (action and column one-hot encodings)
action_types = ['filter', 'group', 'sort', 'projection']
columns = ['transaction_id', 'customer_id', 'amount', 'category', 'region', 'is_returned', 'payment_method']

act_feats = {}
col_feats = {}

for action_id in range(1, 8):
    # One-hot for action type
    act_feats[action_id] = np.zeros(len(action_types))
    act_feats[action_id][action_id % len(action_types)] = 1.0
    
    # One-hot for column
    col_feats[action_id] = np.zeros(len(columns))
    col_feats[action_id][action_id % len(columns)] = 1.0

with open('edge/act_five_feats.pickle', 'wb') as f:
    pickle.dump(act_feats, f)
print('Created edge/act_feats.pickle (also created legacy act_five_feats.pickle for compatibility)')

with open('edge/col_action.pickle', 'wb') as f:
    pickle.dump(col_feats, f, protocol=pickle.HIGHEST_PROTOCOL)

# Create dummy condition features
cond_feats = {}
for action_id in range(1, 8):
    cond_feats[action_id] = np.ones(10) / 10  # Uniform distribution

with open('edge/cond_action.pickle', 'wb') as f:
    pickle.dump(cond_feats, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Created feature pickle files")
```

## Step 3: Run ExplorAct Methods

### Option A: Run EA-SP (Simpler, Faster)

```bash
# Train EA-SP for action-type prediction (τ-rec)
python ../ea_sp.py act 20250212 5 0

# Train EA-SP for column prediction (a-rec)
python ../ea_sp.py col 20250212 5 0

# Train EA-SP for joint action-column prediction (τ,a)-rec)
python ../ea_sp.py tg 20250212 5 0
```

**Parameters explained:**
- `act`: Task type (act=action type, col=column, tg=joint)
- `20250212`: Random seed for reproducibility
- `5`: Context size δ (number of recent actions to consider)
- `0`: Test project ID (0-indexed, 0-3 for 4 projects)

**Output:**
- Results saved to `model_stats/act_20250212_5_[0]_gine_seq.pickle`
- Probabilities saved to `dst_probs/gine_seq_act_best_ra3_20250212_[0]_5.pickle`

### Option B: Run EA-MP (More Accurate but Slower)

```bash
# Train EA-MP for all three tasks with context size 5
python ../ea_mp.py act 20250212 5 0
python ../ea_mp.py col 20250212 5 0
python ../ea_mp.py tg 20250212 5 0
```

This uses multiple perspectives (context trees of sizes 1..5) for better accuracy.

### Option C: Run REACT Baseline

```bash
# Run REACT for comparison
python ../react.py 20250212 5 0
```

## Step 4: Analyze Results

```python
import pickle

# Load EA-SP results
with open('model_stats/act_20250212_5_[0]_gine_seq.pickle', 'rb') as f:
    results = pickle.load(f)
    print("EA-SP Results (τ-rec task):")
    print(f"  Recall@3: {results['ra3']}")  # List of 5 runs
    print(f"  MRR: {results['mrr']}")        # List of 5 runs

# Load DST probability scores for evidence fusion
with open('dst_probs/gine_seq_act_best_ra3_20250212_[0]_5.pickle', 'rb') as f:
    probs = pickle.load(f)
    print(f"Probability scores: {probs}")
```

## Step 5: Run Timing Analysis

To evaluate inference time vs log size:

```bash
# Time inference without training (quick test)
python ../ea_sp_time.py act 5
python ../ea_mp_time.py col 5

# Analyze timing as log size grows
python ../ea_sp_time_logsize.py 20250212 5
```

## Complete Minimal Example

Here's a complete standalone script to create dataset and run a quick test:

```python
#!/usr/bin/env python3
"""Minimal example: create dummy dataset and run ExplorAct."""

import os
import sys
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime, timedelta

def create_dataset():
    """Create dummy dataset structure."""
    os.makedirs('dummy_dataset/raw_datasets', exist_ok=True)
    os.makedirs('dummy_dataset/session_repositories', exist_ok=True)
    os.makedirs('dummy_dataset/display_feats', exist_ok=True)
    os.makedirs('dummy_dataset/edge', exist_ok=True)
    
    os.chdir('dummy_dataset')
    
    # Create raw data
    np.random.seed(42)
    n_records = 500
    df = pd.DataFrame({
        'id': range(1, n_records + 1),
        'value': np.random.exponential(50, n_records),
        'category': np.random.choice(['A', 'B', 'C'], n_records),
        'region': np.random.choice(['X', 'Y', 'Z'], n_records),
    })
    df.to_csv('raw_datasets/1.tsv', sep='\t', index=False)
    
    # Create actions
    actions = pd.DataFrame({
        'action_id': [1, 2, 3],
        'action_type': ['filter', 'group', 'sort'],
        'action_params': [
            json.dumps({"field": "category", "term": "A", "condition": 8}),
            json.dumps({"field": "region", "aggregations": []}),
            json.dumps({"field": "value"})
        ],
        'session_id': [1, 1, 1],
        'user_id': [1, 1, 1],
        'project_id': [1, 1, 1],
        'creation_time': ['2024-01-01 10:00:00', '2024-01-01 10:05:00', '2024-01-01 10:10:00'],
        'parent_display_id': [1, 2, 3],
        'child_display_id': [2, 3, 4],
        'solution': [True, True, True]
    })
    actions.to_csv('session_repositories/actions.tsv', sep='\t', index=False)
    
    # Create displays
    displays = pd.DataFrame({
        'display_id': [1, 2, 3, 4],
        'filtering': [json.dumps({"list": []})] * 4,
        'sorting': [json.dumps({"list": []})] * 4,
        'grouping': [json.dumps({"list": []})]*2 + [json.dumps({"list": [{"field": "region"}]})]*2,
        'aggregations': [None]*4,
        'data_layer': [json.dumps({'id': {'unique': 1.0}, 'value': {'unique': 0.9}, 'category': {'unique': 0.003}, 'region': {'unique': 0.003}})] * 4,
        'granularity_layer': [None]*4,
        'projected_fields': [json.dumps({"list": [{"field": "id"}, {"field": "value"}, {"field": "category"}, {"field": "region"}]})] * 4,
        'session_id': [1, 1, 1, 1],
        'user_id': [1, 1, 1, 1],
        'project_id': [1, 1, 1, 1],
        'solution': [True, True, True, True]
    })
    displays.to_csv('session_repositories/displays.tsv', sep='\t', index=False)
    
    # Create feature files
    display_pca_feats = {i: np.random.randn(181).astype(np.float32) for i in range(1, 5)}
    with open('display_feats/display_pca_feats_9999.pickle', 'wb') as f:
        pickle.dump(display_pca_feats, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    act_feats = {i: np.eye(4)[i % 4] for i in range(1, 4)}
    col_feats = {i: np.eye(4)[i % 4] for i in range(1, 4)}
    cond_feats = {i: np.ones(10) / 10 for i in range(1, 4)}
    
    with open('edge/act_five_feats.pickle', 'wb') as f:
        pickle.dump(act_feats, f)
    with open('edge/col_action.pickle', 'wb') as f:
        pickle.dump(col_feats, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('edge/cond_action.pickle', 'wb') as f:
        pickle.dump(cond_feats, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("✓ Dataset created successfully")
    print("  Raw datasets: raw_datasets/1.tsv")
    print("  Actions: session_repositories/actions.tsv")
    print("  Displays: session_repositories/displays.tsv")
    print("  Features: display_feats/, edge/")

if __name__ == '__main__':
    create_dataset()
    print("\nTo run ExplorAct:")
    print("  python ../ea_sp.py act 20250212 5 0")
```