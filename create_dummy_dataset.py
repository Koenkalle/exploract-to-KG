#!/usr/bin/env python3
"""
Create a self-contained dummy dataset for testing ExplorAct.

This script generates dataset in its own directory without overwriting existing data.
The generated dataset can be selected at runtime by pointing to its directory.

Generated structure:
    my_dataset/
    ├── raw_datasets/1.tsv
    ├── session_repositories/actions.tsv
    ├── session_repositories/displays.tsv
    ├── display_feats/display_pca_feats_*.pickle
    ├── edge/{act_five_feats,col_action,cond_action}.pickle
    └── chunked_sessions/unbiased_seed_*.pickle

Usage:
    python create_dummy_dataset.py --output my_dataset
    
Then use interchangeably with existing dataset:
    # Use dummy dataset
    python -c "import sys; sys.path.insert(0, 'my_dataset'); from lib import Repository; repo = Repository(...)"
    
    # Or use existing dataset (default)
    python ea_sp.py act 20250212 5 0  # Uses default session_repositories/
"""

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


def create_raw_dataset(output_dir: Path, n_records: int = 1000) -> None:
    """Create raw dataset TSV file."""
    raw_dir = output_dir / 'raw_datasets'
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    
    # Generate e-commerce transaction data
    df = pd.DataFrame({
        'transaction_id': range(1, n_records + 1),
        'customer_id': np.random.randint(1, 100, n_records),
        'amount': np.random.exponential(scale=50, size=n_records),
        'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n_records),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_records),
        'date': [datetime(2024, 1, 1) + timedelta(days=int(d)) 
                 for d in np.random.exponential(scale=30, size=n_records)],
        'is_returned': np.random.choice([True, False], n_records, p=[0.1, 0.9]),
        'payment_method': np.random.choice(['Credit', 'Debit', 'PayPal'], n_records)
    })
    
    dataset_path = raw_dir / '1.tsv'
    df.to_csv(dataset_path, sep='\t', index=False)
    print(f"✓ Created raw dataset: {dataset_path} ({df.shape[0]} records, {df.shape[1]} columns)")
    
    return df


def create_actions_log(output_dir: Path) -> None:
    """Create session actions log."""
    session_dir = output_dir / 'session_repositories'
    session_dir.mkdir(parents=True, exist_ok=True)
    
    actions_data = []
    action_id = 1
    
    # Define 3 different user sessions with different patterns
    session_patterns = [
        # Session 1: Filter Electronics -> Group by Region -> Sort by Amount
        [
            {
                'type': 'filter',
                'field': 'category',
                'params': {"field": "category", "term": "Electronics", "condition": 8}
            },
            {
                'type': 'group',
                'field': 'region',
                'params': {"field": "region", "aggregations": []}
            },
            {
                'type': 'sort',
                'field': 'amount',
                'params': {"field": "amount"}
            }
        ],
        # Session 2: Filter by Region -> Group by Category -> Aggregate
        [
            {
                'type': 'filter',
                'field': 'region',
                'params': {"field": "region", "term": "North", "condition": 8}
            },
            {
                'type': 'group',
                'field': 'category',
                'params': {"field": "category", "aggregations": [{"field": "amount", "type": "sum"}]}
            },
            {
                'type': 'sort',
                'field': 'amount',
                'params': {"field": "amount"}
            }
        ],
        # Session 3: Filter Payment -> Filter Amount -> Group Customer
        [
            {
                'type': 'filter',
                'field': 'payment_method',
                'params': {"field": "payment_method", "term": "Credit", "condition": 8}
            },
            {
                'type': 'filter',
                'field': 'amount',
                'params': {"field": "amount", "term": "100", "condition": 32}
            },
            {
                'type': 'group',
                'field': 'customer_id',
                'params': {"field": "customer_id", "aggregations": [{"field": "amount", "type": "count"}]}
            }
        ]
    ]
    
    # Create actions for multiple users, sessions, and projects
    # Use projects 0-3 to allow leave-one-out cross-validation (standard in EA paper)
    # Generate enough data per project to have meaningful train/test splits
    # Keep display_id < 427 to avoid logic_error_displays hardcoded in ea_sp.py
    display_id = 1
    for project_id in range(0, 4):  # 4 projects (0-3) for cross-validation
        # Create 3 repetitions of sessions per project
        for rep in range(3):
            for session_id, pattern in enumerate(session_patterns, start=1):
                for user_id in range(1, 3):  # 2 users per session
                    for action_idx, action in enumerate(pattern, start=1):
                        actions_data.append({
                            'action_id': action_id,
                            'action_type': action['type'],
                            'action_params': json.dumps(action['params']),
                            'session_id': session_id + (project_id - 1) * 100 + rep * 10,  # Unique session IDs
                            'user_id': user_id + (project_id - 1) * 100,
                            'project_id': project_id,
                            'creation_time': f'2024-01-0{(action_idx % 9) + 1} {10+action_idx:02d}:00:00',
                            'parent_display_id': display_id,
                            'child_display_id': display_id + 1,
                            'solution': True
                        })
                        display_id += 1
                        action_id += 1
    
    actions_df = pd.DataFrame(actions_data)
    actions_path = output_dir / 'session_repositories' / 'actions.tsv'
    actions_df.to_csv(actions_path, sep='\t', index=False)
    print(f"✓ Created actions log: {actions_path} ({len(actions_data)} actions)")


def create_displays_log(output_dir: Path) -> None:
    """Create session displays snapshots."""
    session_dir = output_dir / 'session_repositories'
    session_dir.mkdir(parents=True, exist_ok=True)
    
    displays_data = []
    
    # Base data layer statistics
    base_data_layer = {
        'transaction_id': {'unique': 1.0, 'entropy': 1.0},
        'customer_id': {'unique': 0.1, 'entropy': 0.5},
        'amount': {'unique': 0.95, 'entropy': 0.9},
        'category': {'unique': 0.004, 'entropy': 0.15},
        'region': {'unique': 0.004, 'entropy': 0.15},
        'is_returned': {'unique': 0.002, 'entropy': 0.1},
        'payment_method': {'unique': 0.003, 'entropy': 0.12},
        'date': {'unique': 0.99, 'entropy': 0.95}
    }
    
    # Create displays for each session and action step
    display_id = 1
    session_patterns = [
        [('category', ['region'], []),
         ('region', ['region'], ['amount']),
         ('amount', ['region'], ['amount'])],
        
        [('region', ['region'], []),
         ('category', ['category'], ['amount']),
         ('amount', ['category'], ['amount'])],
        
        [('payment_method', ['payment_method'], []),
         ('amount', ['payment_method'], ['amount']),
         ('customer_id', ['customer_id'], ['amount'])]
    ]
    
    for session_id, pattern in enumerate(session_patterns, start=1):
        for user_id in range(1, 3):  # 2 users per session
            # Initial unfiltered display
            displays_data.append({
                'display_id': display_id,
                'filtering': json.dumps({"list": []}),
                'sorting': json.dumps({"list": []}),
                'grouping': json.dumps({"list": []}),
                'aggregations': None,
                'data_layer': json.dumps(base_data_layer),
                'granularity_layer': None,
                'projected_fields': json.dumps({"list": [{"field": f} for f in base_data_layer.keys()]}),
                'session_id': session_id,
                'user_id': user_id,
                'project_id': 1,
                'solution': True
            })
            display_id += 1
            
            # Display after each action
            for filter_field, group_fields, sort_fields in pattern:
                granularity = {'size_mean': 100, 'group_attrs': group_fields} if group_fields else None
                displays_data.append({
                    'display_id': display_id,
                    'filtering': json.dumps({"list": []}),
                    'sorting': json.dumps({"list": [{"field": f} for f in sort_fields]}),
                    'grouping': json.dumps({"list": [{"field": f, "groupPriority": i} for i, f in enumerate(group_fields)]}),
                    'aggregations': json.dumps({"list": []}) if group_fields else None,
                    'data_layer': json.dumps(base_data_layer),
                    'granularity_layer': json.dumps(granularity) if granularity else None,
                    'projected_fields': json.dumps({"list": [{"field": f} for f in base_data_layer.keys()]}),
                    'session_id': session_id,
                    'user_id': user_id,
                    'project_id': 1,
                    'solution': True
                })
                display_id += 1
    
    displays_df = pd.DataFrame(displays_data)
    displays_path = output_dir / 'session_repositories' / 'displays.tsv'
    displays_df.to_csv(displays_path, sep='\t', index=False)
    print(f"✓ Created displays log: {displays_path} ({len(displays_data)} displays)")


def create_feature_files(output_dir: Path) -> None:
    """Create dummy feature pickle files."""
    display_dir = output_dir / 'display_feats'
    edge_dir = output_dir / 'edge'
    display_dir.mkdir(parents=True, exist_ok=True)
    edge_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the generated data to know actual sizes
    actions_df = pd.read_csv(output_dir / 'session_repositories' / 'actions.tsv', sep='\t')
    n_actions = len(actions_df)
    
    # Calculate number of displays: initial display + one per action + final displays
    # 4 projects × 3 sessions × 2 users × 3 actions per session + 1 initial per session
    n_displays = max(actions_df['child_display_id'].max(), actions_df['parent_display_id'].max()) + 10
    
    # Create dummy node features (181-dimensional, matching paper)
    np.random.seed(42)
    display_pca_feats = {i: np.random.randn(181).astype(np.float32) for i in range(1, n_displays + 1)}
    
    display_feats_path = display_dir / 'display_pca_feats_9999.pickle'
    with open(display_feats_path, 'wb') as f:
        pickle.dump(display_pca_feats, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Created node features: {display_feats_path} ({len(display_pca_feats)} displays × 181 dims)")
    
    # Create dummy edge features
    # Note: The model concatenates act_feats + col_feats to create concat_feats (line 80 ea_sp.py)
    # concat_feats is used as edge attributes; EDGE_DIM is computed dynamically from the data
    # So: act_feats (12 dims) + col_feats (8 dims) = 20 dims total for this dummy dataset
    action_types = ['filter', 'group', 'sort', 'projection']
    columns = ['transaction_id', 'customer_id', 'amount', 'category', 'region', 'is_returned', 'payment_method', 'date']
    
    # act_feats will be 12-dimensional (4 action types one-hot + 8 padding)
    # col_feats will be 8-dimensional (8 columns one-hot)
    # Total after concatenation: 12 + 8 = 20 dimensions
    
    act_feats = {}
    col_feats = {}
    cond_feats = {}
    
    for action_id in range(1, n_actions + 1):
        # Create 12-dimensional action features (one-hot action type + padding)
        act_feats[action_id] = np.zeros(12, dtype=np.float32)
        
        # One-hot for action type (first 4 positions)
        action_type_idx = action_id % len(action_types)
        act_feats[action_id][action_type_idx] = 1.0
        # Rest (positions 4-11) are padding - leave as zeros
        
        # Create 8-dimensional column features (one-hot for each column)
        col_feats[action_id] = np.zeros(len(columns), dtype=np.float32)
        col_idx = action_id % len(columns)
        col_feats[action_id][col_idx] = 1.0
        
        # Dummy condition features (uniform distribution)
        cond_feats[action_id] = (np.ones(10, dtype=np.float32) / 10)
    
    with open(edge_dir / 'act_five_feats.pickle', 'wb') as f:
        pickle.dump(act_feats, f, protocol=pickle.HIGHEST_PROTOCOL)
    # Also write the new-name file for compatibility
    with open(edge_dir / 'act_feats.pickle', 'wb') as f:
        pickle.dump(act_feats, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Created action features: {edge_dir / 'act_feats.pickle'} and legacy {edge_dir / 'act_five_feats.pickle'} ({len(act_feats)} actions, 12 dims)")
    
    with open(edge_dir / 'col_action.pickle', 'wb') as f:
        pickle.dump(col_feats, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Created column features: {edge_dir / 'col_action.pickle'} ({len(col_feats)} actions, 8 dims)")
    
    with open(edge_dir / 'cond_action.pickle', 'wb') as f:
        pickle.dump(cond_feats, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Created condition features: {edge_dir / 'cond_action.pickle'} ({len(cond_feats)} actions)")


def create_wrapper_script(output_dir: Path) -> None:
    """Create a simple wrapper script to run EA-SP from the dataset directory."""
    wrapper_content = '''#!/usr/bin/env python3
"""Wrapper script to run ExplorAct methods from dataset directory."""

import sys
import os
import subprocess

# Get the path to the exploract root (parent of this directory)
dataset_dir = os.path.dirname(os.path.abspath(__file__))
exploract_root = os.path.dirname(dataset_dir)

# Change to exploract root for imports to work
os.chdir(exploract_root)

# Add BOTH exploract root and dataset directory to Python path
sys.path.insert(0, exploract_root)
sys.path.insert(0, dataset_dir)

# Import and patch the paths in the ea_sp module before it loads files
if __name__ == '__main__':
    # Modify sys.argv to use absolute paths
    script_name = sys.argv[1] if len(sys.argv) > 1 else 'ea_sp'
    script_path = os.path.join(exploract_root, f'{script_name}.py')
    
    if not os.path.exists(script_path):
        print(f"Error: {script_path} not found")
        sys.exit(1)
    
    # Adjust sys.argv: remove script name, keep remaining args
    # sys.argv[0] stays as the script path, sys.argv[1:] are the actual arguments
    sys.argv = [script_path] + sys.argv[2:]
    
    # Read the script
    with open(script_path, 'r') as f:
        script_code = f.read()
    
    # Replace relative paths with dataset-aware paths
    script_code = script_code.replace(
        "repo = Repository('./session_repositories/actions.tsv','./session_repositories/displays.tsv','./raw_datasets/')",
        f"repo = Repository('{dataset_dir}/session_repositories/actions.tsv','{dataset_dir}/session_repositories/displays.tsv','{dataset_dir}/raw_datasets/')"
    )
    
    # Replace other file path references
    old_paths = [
        ("'./chunked_sessions/", f"'{dataset_dir}/chunked_sessions/"),
        ("'./edge/", f"'{dataset_dir}/edge/"),
        ("'./display_feats/", f"'{dataset_dir}/display_feats/"),
        ("'./model_stats/", f"'{dataset_dir}/model_stats/"),
        ("'./dst_probs/", f"'{dataset_dir}/dst_probs/"),
    ]
    
    for old_path, new_path in old_paths:
        script_code = script_code.replace(old_path, new_path)
    
    # Create model_stats and dst_probs directories if they don't exist
    os.makedirs(f'{dataset_dir}/model_stats', exist_ok=True)
    os.makedirs(f'{dataset_dir}/dst_probs', exist_ok=True)
    
    # Execute the modified script with proper globals
    exec_globals = {
        '__file__': script_path,
        '__name__': '__main__',
        '__builtins__': __builtins__,
    }
    try:
        exec(script_code, exec_globals)
    except SystemExit:
        pass
'''
    
    wrapper_path = output_dir / 'run.py'
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_content)
    os.chmod(wrapper_path, 0o755)
    print(f"✓ Created wrapper script: {wrapper_path}")


def create_symlinks(output_dir: Path) -> None:
    """Create symlinks to exploract source files so imports work."""
    parent_dir = output_dir.parent
    
    # Symlink lib directory
    lib_link = output_dir / 'lib'
    lib_source = parent_dir / 'lib'
    if lib_source.exists() and not lib_link.exists():
        try:
            lib_link.symlink_to(lib_source)
            print(f"✓ Symlinked lib directory")
        except Exception as e:
            print(f"  Warning: Could not symlink lib: {e}")


def create_session_chunks(output_dir: Path) -> None:
    """Create chunked session splits (for cross-validation).
    
    Structure: {project_id: [edge_list_session1, edge_list_session2, ...]}
    where each edge_list is the sequence of [parent_display_id, child_display_id, {'aid': action_id}] triplets.
    """
    chunks_dir = output_dir / 'chunked_sessions'
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    # Read actions and displays to build edges
    actions_df = pd.read_csv(output_dir / 'session_repositories' / 'actions.tsv', sep='\t')
    
    # Build edge lists grouped by project and session
    # Each edge must be [parent_display_id, child_display_id, {'aid': action_id}] to match ea_sp.py expectations
    chunks_by_project = {}
    
    for project_id in actions_df['project_id'].unique():
        project_actions = actions_df[actions_df['project_id'] == project_id]
        sessions = []
        
        for session_id in project_actions['session_id'].unique():
            session_actions = project_actions[
                project_actions['session_id'] == session_id
            ].sort_values('action_id')
            
            # Build edge list for this session with aid attribute
            edges = []
            for _, row in session_actions.iterrows():
                parent = int(row['parent_display_id'])
                child = int(row['child_display_id'])
                action_id = int(row['action_id'])
                # Edge format: [u, v, {'aid': aid}] as expected by ea_sp.py line 321
                edges.append([parent, child, {'aid': action_id}])
            
            if edges:  # Only add if session has actions
                sessions.append(edges)
        
        if sessions:  # Only add if project has sessions
            chunks_by_project[project_id] = sessions
    
    # Create session chunks for different seeds
    for seed in range(20250212, 20250217):  # 5 different seeds
        chunk_path = chunks_dir / f'unbiased_seed_{seed}.pickle'
        with open(chunk_path, 'wb') as f:
            pickle.dump(chunks_by_project, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"✓ Created session chunks: {chunks_dir}/unbiased_seed_*.pickle (5 seeds)")
    print(f"  Structure: {{project_id: [edge_lists]}}")
    for proj_id, sessions in chunks_by_project.items():
        print(f"    Project {proj_id}: {len(sessions)} sessions")


def main():
    parser = argparse.ArgumentParser(
        description='Create self-contained dummy dataset for ExplorAct',
        epilog='''
EXAMPLES:
  # Create dummy dataset in isolation
  python create_dummy_dataset.py --output my_test_data
  
  # Run ExplorAct on dummy dataset (creates model_stats/ inside my_test_data/)
  cd my_test_data && python ../ea_sp.py act 20250212 5 0
  
  # Use existing dataset (default in cwd)
  cd . && python ea_sp.py act 20250212 5 0  # Uses session_repositories/ in cwd
  
  # Create multiple independent datasets for testing
  python create_dummy_dataset.py --output dataset_v1 --records 500
  python create_dummy_dataset.py --output dataset_v2 --records 2000
        '''
    )
    parser.add_argument('--output', type=str, default='.', 
                       help='Output directory for self-contained dataset (default: current directory)')
    parser.add_argument('--records', type=int, default=1000,
                       help='Number of records in raw dataset (default: 1000)')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Creating self-contained dummy dataset in: {output_dir.resolve()}")
    print(f"{'='*70}\n")
    
    create_raw_dataset(output_dir, n_records=args.records)
    create_actions_log(output_dir)
    create_displays_log(output_dir)
    create_feature_files(output_dir)
    create_session_chunks(output_dir)
    create_symlinks(output_dir)
    create_wrapper_script(output_dir)
    
    print(f"\n{'='*70}")
    print("✓ Dataset creation complete!")
    print(f"{'='*70}\n")
    
    # Get the absolute path for clear instructions
    abs_path = output_dir.resolve()
    exploract_root = abs_path.parent
    
    print("USAGE OPTIONS:\n")
    print(f"Option 1 - Use the wrapper script (RECOMMENDED):")
    print(f"  cd {abs_path}")
    print(f"  python run.py ea_sp act 20250212 5 0")
    print(f"  python run.py ea_mp col 20250212 5 0")
    print(f"  python run.py react 20250212 5 0")
    print(f"  → Results saved to: {abs_path}/model_stats/")
    print(f"\nOption 2 - Run from exploract root (pass dataset path):")
    print(f"  cd {exploract_root}")
    print(f"  DATASET_PATH={abs_path} python ea_sp.py act 20250212 5 0")
    print(f"\nOption 3 - Use existing dataset (default in cwd):")
    print(f"  cd {exploract_root}")
    print(f"  python ea_sp.py act 20250212 5 0  # Uses ./session_repositories/")
    print(f"\nDataset structure created:")
    print(f"  {abs_path}/")
    print(f"  ├── raw_datasets/1.tsv")
    print(f"  ├── session_repositories/")
    print(f"  │   ├── actions.tsv")
    print(f"  │   └── displays.tsv")
    print(f"  ├── display_feats/")
    print(f"  │   └── display_pca_feats_*.pickle")
    print(f"  ├── edge/")
    print(f"  │   ├── act_five_feats.pickle")
    print(f"  │   ├── col_action.pickle")
    print(f"  │   └── cond_action.pickle")
    print(f"  └── chunked_sessions/")
    print(f"      └── unbiased_seed_*.pickle")
    print()


if __name__ == '__main__':
    main()
