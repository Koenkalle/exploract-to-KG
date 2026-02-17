#!/usr/bin/env python3
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
