#!/bin/bash
# Wrapper script to run ExplorAct models from wikidata_dataset directory
# 
# Usage: ./run.py <model> <task> <seed> <main_size> <test_id>
# Examples:
#   python run.py ea_sp act 20250212 5 0
#   python run.py ea_mp col 20250212 5 0
#   python run.py react 20250212 5 0

import sys
import os

# Add parent directory to path so we can import from parent exploract dir
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Get the script name and arguments
if len(sys.argv) < 2:
    print("Usage: python run.py <model> <task> <seed> <main_size> <test_id>")
    print()
    print("Models: ea_sp, ea_mp, react")
    print("Tasks: act (action type), col (column), tg (action-column joint)")
    print()
    print("Examples:")
    print("  python run.py ea_sp act 20250212 5 0")
    print("  python run.py ea_mp col 20250212 5 0")
    sys.exit(1)

model_name = sys.argv[1]
task = sys.argv[2]
seed = sys.argv[3]
main_size = sys.argv[4]
test_id = sys.argv[5]

# Map model names to actual script files
model_scripts = {
    'ea_sp': 'ea_sp.py',
    'ea_mp': 'ea_mp.py',
    'react': 'react.py',
}

if model_name not in model_scripts:
    print(f"Unknown model: {model_name}")
    print(f"Available models: {', '.join(model_scripts.keys())}")
    sys.exit(1)

script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                          model_scripts[model_name])

if not os.path.exists(script_path):
    print(f"Error: Script not found at {script_path}")
    sys.exit(1)

# Construct argv as if the model script was called directly
sys.argv = [script_path, task, seed, main_size, test_id]

# Execute the model script
with open(script_path, 'r') as f:
    code = f.read()
    exec(code)
