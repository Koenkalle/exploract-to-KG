````instructions
# ExplorAct Codebase - AI Agent Instructions

## Research Paper Context

This codebase implements **ExplorAct: Context-Aware Next Action Recommendations for Interactive Data Exploration** (CIKM 2025, DOI: `10.1145/3746252.3761257`). 

The paper addresses a critical problem in modern interactive data analysis (IDA) platforms (Tableau, Power BI, Looker, Kibana, Splunk): **predicting the next action users will take during exploratory analysis**. 

### Key Innovation
Unlike existing SOTA methods (REACT) that use simple k-NN with tree edit distance (TED) and suffer from **log-size-dependent retrieval**, ExplorAct achieves **constant-time inference** by combining:
1. **Graph Isomorphism Networks (GINE)** to capture structural context of exploration sessions
2. **Gated Recurrent Units (GRU)** to model temporal sequences of actions
3. **Dempster-Shafer Theory (DST)** for evidence fusion across multiple recommendation tasks

### Experimental Results
- **Action-Type Recommendation (τ-rec)**: +16.98% Recall@3 (11.5% avg improvement)
- **Column Recommendation (a-rec)**: +22.22% Recall@3 (17.02% avg), up to 34.38% MRR improvement
- **Joint Action-Column (τ,a)-rec**: +20.94% Recall@3 (13.69% avg), up to 13.87% MRR improvement
- **Inference**: Constant time regardless of log size (depends only on max context size)

## Project Overview

**ExplorAct** is a machine learning framework for graph-based sequence recommendation in exploratory data analysis. It reconstructs user interaction sessions as directed acyclic graphs (DAGs) and uses graph neural networks (GINE, GRU) to predict the next action in a query session.

### Key Concepts (From Paper)
- **Analysis Tree (Ψ)**: Hierarchical DAG representing a user's entire exploration session
  - Nodes: Action results (Δᵣ) at each step
  - Edges: Analysis actions (α) that transform one result to next
  - Root: Initial dataset
- **δ-Context Tree (ψ)**: Minimal subtree containing δ most recent nodes before an action
  - Captures immediate exploration context leading to next action
  - Paper uses δ ∈ {3,4,5,6,7,8} for experiments
- **Action**: 5-tuple (τ, a, κ, ω, Λ) where:
  - **τ**: Action type ∈ {Projection, Filter, Group, Sort}
  - **a**: Column/attribute being operated on
  - **κ**: Operator (=, ≤, >, etc.) - optional
  - **ω**: Filter values - optional
  - **Λ**: Aggregation specs (count, min, max, sum, avg) - optional
- **Prediction Tasks** (paper notation):
  - **τ-rec (act)**: Action type prediction only
  - **a-rec (col)**: Column prediction only
  - **(τ, a)-rec (tg)**: Joint action-type and column prediction
- **Candidate Representative Graph**: Average graph structure aggregating all context trees for a candidate
  - Captures common patterns within each action type/column class
  - Nodes labeled with positional identities (depth, index within depth)
- **Sequence Formulation**:
  - **State-Perspective (EA-SP)**: Single sequence of δ-context trees at each step
  - **Multi-Perspective (EA-MP)**: Sequence of context trees at sizes 1..δ (oldest to newest)

## Architecture & Data Flow

### Three Main Variants

1. **EA-SP (Expand-Analysis Single Path)** - `ea_sp.py`, `ea_sp_time.py`, `ea_sp_time_logsize.py`
   - Uses **State-Perspective context tree sequence**: single δ-context tree per action step
   - Each step represents user's state at that action
   - Architecture: GINE(query) → Interaction Features (diff, hadamard, concat) → GRU → Sigmoid classifier
   - **Paper results**: Baseline GNN approach, good accuracy/speed tradeoff
   - **Complexity**: O(δ × |GRU|) where δ is sequence length (context size)

2. **EA-MP (Expand-Analysis Multi Path)** - `ea_mp.py`, `ea_mp_time.py`, `ea_mp_time_logsize.py`
   - Uses **Multi-Perspective context tree sequence**: multiple trees of sizes 1..δ per action
   - Ordered oldest (size 1) to newest (size δ) for temporal perspective
   - Captures exploration context at multiple granularities
   - **Paper results**: +16.98% improvement vs SOTA on action-type (τ-rec) task
   - **Complexity**: O(δ² × |GRU|) but with improved accuracy

3. **REACT** - `react.py`, `react_time_logsize.py`
   - **State-of-the-art baseline** from prior work (REACT-IDA benchmark)
   - Non-learning k-NN approach using custom tree edit distance (TED)
   - Uses BallTree for nearest neighbor search
   - **Paper comparison**: Fast but limited accuracy, log-size-dependent retrieval cost
   - **Complexity**: O(n log n) nearest neighbor search on training set

### Core Data Processing Pipeline

```
Session Edges (TSV) → replay_graph() → Tree Extraction → Feature Assignment
       ↓                  ↓                    ↓                   ↓
session_repositories/   Extract context    BFS traversal      Node/Edge attrs
actions/displays        around endpoints   assign depth/pos     (pos, x)
                        
                        ↓
                    treefy_sessions() → PyG Conversion → Model Training
                    (train/test split)
```

## Paper's Core Contributions

The paper makes four key contributions:

1. **Problem Formulation**: Frames next-action recommendation as candidate probability prediction
   - Enables training on new candidates incrementally without restructuring model
   - Supports three recommendation tasks: τ-rec, a-rec, (τ,a)-rec

2. **Novel Deep Learning Architecture**: Hybrid GINE + GRU model
   - **GINE** for structural context: captures graph isomorphism properties of exploration trees
   - **GRU** for sequential patterns: models temporal dynamics of action sequences
   - Two input encoding strategies (State-Perspective EA-SP, Multi-Perspective EA-MP)

3. **Evidence Fusion via Dempster-Shafer Theory (DST)**
   - Each task model (τ-rec, a-rec, (τ,a)-rec) acts as independent evidence source
   - DST combines sources accounting for uncertainty (belief and plausibility)
   - Results in more reliable joint action-type and column recommendations (EF-SP, EF-MP)
   - See `evidence_fusion.ipynb` for implementation

4. **Experimental Validation**: Comprehensive real-world evaluation
   - **Datasets**: REACT-IDA benchmark (4 cybersecurity datasets, 56 analysts)
   - **Metrics**: Recall@k, Mean Reciprocal Rank across three recommendation tasks
   - **Key finding**: Constant-time inference (vs log-size-dependent REACT)

## Critical Functions (Review These First)

- **`replay_graph()`** (~100 lines): Core logic to extract context graphs from session edges
  - Filters out `logic_error_displays` (invalid endpoint nodes)
  - Builds tree_nodes up to `main_size` nodes
  - Uses "backtracking" algorithm to find common ancestor paths
  - Returns: graphs, action sequences, labels, ending action IDs
  
- **`treefy_sessions()`**: Orchestrates replay_graph across train/test splits
  
- **`generate_lump_graphs()`**: Aggregates node/edge features by depth and position to create class prototypes
  
- **`generate_tree_sequences_train/test()`**: Maps action IDs to trees and creates torch datasets
  
- **`MatchingNetwork`**: GNN encoder with LSTM matching head for sequence ranking

## File Organization

```
exploract/
├── ea_sp.py              # Main EA-SP accuracy model (910 lines)
├── ea_sp_time.py         # EA-SP inference timing (no model train)
├── ea_sp_time_logsize.py # EA-SP with varying log sizes
├── ea_mp.py              # Multi-path variant (876 lines)
├── ea_mp_time*.py        # MP timing variants
├── react.py              # TED-based baseline (629 lines)
├── naive_gine.py         # GINE without sequences
├── naive_gru.py          # GRU without sequences
├── lib/
│   ├── utilities.py      # Repository class (TSV pandas wrapper)
│   └── distance.py       # Action/display distance metrics
├── {edge,display_feats}/ # Pre-computed feature pickles
│   ├── act_feats.pickle (or legacy act_five_feats.pickle)
│   ├── col_action.pickle
│   ├── cond_action.pickle
│   └── display_pca_feats_*.pickle
├── chunked_sessions/     # Session edge lists (unbiased_seed_*.pickle)
├── result_analyser.ipynb # Accuracy/plot generation
├── evidence_fusion.ipynb # EF-SP/EF-MP evaluation
└── [node,edge]_feat_gen.ipynb # Feature engineering
```

## Key Conventions

### Parameters & Naming
- **`seed`**: Random seed for reproducibility (integer)
- **`main_size`**: Primary context size for trees (1-10 typical)
- **`min_size`**: Minimum nodes to keep a sample (usually 1)
- **`test_id`**: List of project IDs to hold out (e.g., `[4]` for 4-way split)
- **`task`**: `'act'`, `'col'`, or `'tg'` determines feature set (line 695+)

## Node and Edge Feature Engineering (Paper Section 3.2)

### Node Features (Section 3.2.1-3.2.4)
For each column in an action result, generate probability vectors based on column type:

1. **Numerical Columns**: 
   - Discretize into equal-width bins over range [min_a(D), max_a(D) + ε)
   - Create probability vector over bins: P_a = [P(B_1|a(Δ_r)), P(B_2|a(Δ_r)), ...]
   - Aggregated forms (avg, min, max, sum) use different ranges
   - Count aggregations use [0, max(|a(D_i)|) + ε)

2. **Categorical Columns**:
   - Extract unique values Q from column
   - Create probability vector: P_a = [P(q_1|a(Δ_r)), P(q_2|a(Δ_r)), ...]
   - Applied to original categorical and min/max aggregations of ordinal columns

3. **Text Columns**:
   - Extract finite templates using rule-based algorithm + regex
   - Create probability vector: P_a = [P(t_1|a(Δ_r)), P(t_2|a(Δ_r)), ...]

4. **Final Node Feature**:
   - Concatenate all column probability vectors: v(Δ_r) = ⊕_{a∈C} P_a
   - Apply PCA to reduce to top d_v' components
   - Result: node feature vector v(u_r) = v'(Δ_r) with 181 dimensions (paper uses d_v'=181)

### Edge Features (Section 3.2.5)
- One-hot encode action type τ
- One-hot encode column a
- Concatenate: v_α = [one_hot(τ) ⊕ one_hot(a)]
- Final dimension: **dynamically computed** (REACT-IDA: 20, Wikidata: 60)

### Dataset Feature Dimensions Quick Reference

| Feature | REACT-IDA | Wikidata |
|---|---|---|
| Node dim (`display_pca_feats`) | 181 | 181 |
| Action types (`act_feats`) | 4 | 4 |
| Columns (`col_feats`) | 14 | 56 |
| Edge dim (`concat_feats`) | 20 | 60 |

### Model Function (Paper Section 3.5)
The model uses a matching function to compute likelihood of context tree sequence leading to action:

1. **GINE Processing** (Eq. 1): 
   - h^(l)_u = σ(BN(MLP((1+ε)h^(l-1)_u + Σ_{v∈N(u)} σ(h^(l-1)_v + W·h_uv))))
   - Global add pooling concatenates all layer outputs
   - Linear projection + Leaky ReLU produces graph embedding

2. **Interaction Features** (Eq. 4):
   - For each context tree in sequence: x_ψ = x'_ψ ⊕ (x'_ψ - x_ζ) ⊕ (x'_ψ ⊙ x_ζ)
   - Three components: original, difference, hadamard product
   - Captures both agreement and divergence between query and candidate

3. **GRU Sequencing** (Eq. 5):
   - Stack of GRU layers processes interaction features temporally
   - Update and reset gates control information flow
   - Final hidden states concatenated: h⊕_T = ⊕^L_{l=1} h^(l)_T

4. **Final Likelihood** (Eq. 5):
   - Combine candidate graph embedding x_ζ with GRU output: h = x_ζ ⊕ h⊕_T
   - Linear layer + Sigmoid: p = Sigmoid(W·h + b) ∈ [0,1]
   - Output: probability that sequence leads to candidate action

### Seeding & Reproducibility
- Every training run calls `seed_everything()` with OS-based random seed
- Ensures different random initializations across 5 runs per config
- Results saved to `model_stats/` and `dst_probs/` with seed/test_id in filename

### Device Management
- Default: `torch.device('cuda')` at top of each script
- **To switch to CPU**: Edit line ~45 in ea_sp.py, ea_mp.py: `device = torch.device('cpu')`

## Running Experiments

### Training (Accuracy)
```bash
# EA-SP: predict task 'act', seed 0, context size 5, test on project 3
python ea_sp.py act 0 5 3

# EA-MP: same but multi-path variant
python ea_mp.py col 0 5 3

# REACT: baseline
python react.py 0 5 3
```
Output: `model_stats/{task}_{seed}_{main_size}_{test_id}_*.pickle`

### Inference Timing (Single Run)
```bash
python ea_sp_time.py act 5    # No training, just time inference
python ea_mp_time.py col 5
```

### Scaling Analysis
```bash
python ea_sp_time_logsize.py 0 5  # Timing across log sizes
```

## Quick Start - Running on New Datasets

**Datasets are self-contained and interchangeable** — create independently without affecting existing data.

**See also:** 
- `RUNNING_ON_NEW_DATASET.md` - Comprehensive step-by-step guide with code examples
- `QUICK_REFERENCE.md` - Condensed command reference and file structure reference
- `create_dummy_dataset.py` - Automated script to generate self-contained dummy dataset

### 5-Minute Setup (Two Approaches)

**Approach 1: Use Default Dataset**
```bash
# From exploract root, uses ./session_repositories/ (existing data)
python ea_sp.py act 20250212 5 0
```

**Approach 2: Create and Use Isolated Dummy Dataset**
```bash
# 1. Generate self-contained dummy dataset (doesn't affect existing data)
python create_dummy_dataset.py --output my_dataset

# 2. Run ExplorAct from within it
cd my_dataset
python ../ea_sp.py act 20250212 5 0

# 3. Results isolated to my_dataset/model_stats/
ls model_stats/
```

**Approach 3: Create Multiple Independent Datasets**
```bash
# Create versions for different experiments (all isolated)
python create_dummy_dataset.py --output small_test --records 500
python create_dummy_dataset.py --output large_test --records 5000

# Run each independently, no conflicts
cd small_test && python ../ea_sp.py act 20250212 5 0
cd ../large_test && python ../ea_sp.py col 20250212 5 0
```

### Dataset Requirements

Each self-contained dataset needs three components:
1. **Raw data** (`raw_datasets/1.tsv`): Tab-separated table with multiple columns
2. **Action log** (`session_repositories/actions.tsv`): User interaction history
3. **Display snapshots** (`session_repositories/displays.tsv`): State after each action

### Feature Files Required
4. **Node features** (`display_feats/display_pca_feats_*.pickle`): 181-dimensional embeddings per display
5. **Edge features** (`edge/act_feats.pickle` (or legacy `act_five_feats.pickle`), `col_action.pickle`, `cond_action.pickle`): One-hot encodings
6. **Session chunks** (`chunked_sessions/unbiased_seed_*.pickle`): Train/test splits for cross-validation

See `QUICK_REFERENCE.md` for exact file format specifications and examples.

## Common Patterns & Gotchas

1. **Logic Error Displays**: Hardcoded list (line ~100) filters invalid endpoints. **Don't remove** without domain validation.

2. **Feature Concatenation**: 
   - `concat_feats[action_id]` = action_feats + col_feats (separate indices)
   - Not actcol_feats (which uses argmax to create one-hot)

3. **One-Hot Edge Features**: 
   ```python
   offset = np.argmax(act_feats[key]) * len(col_feats[key])
   feat[offset + np.argmax(col_feats[key])] = 1  # Only one position set
   ```

4. **Batch Size Logic**: Adjusted to avoid size-1 batches
   ```python
   while train_length % batch_size == 1:
       batch_size += 1
   ```

5. **Train/Test Split**: Controlled by `chunked_sessions` pickle which pre-partitions by seed. Only `test_id` projects held out; others are train.

6. **MultiPath (EA-MP) Specifics**:
   - `sizes = list(range(1, main_size + 1))` then reversed
   - Generates pair of sequences (display_seqs, action_seqs)
   - `generate_train_pairs()` creates cross-class positives/negatives

7. **Graph Directionality**: 
   - Input: undirected from session edges
   - Converted: `to_directed()` then remove edges where `source > target`
   - Result: DAG preserving temporal ordering

## Editing & Extending

### Adding a New Feature Type
1. Load in feature pickle (lines 30-40)
2. Register in task selection (line ~695)
3. Pass to `treefy_sessions()` as `tar=feature`

### Modifying Graph Construction
- **`replay_graph()`** is where context selection happens (lines 150-290)
- Node count controlled by `main_size` loop
- Backtrack algorithm finds common ancestry; edit lines 200-230 to change

### Changing Model Architecture
- **`MatchingNetwork.__init__()`**: GIN/GAT layer counts, hidden dims
- **`make_gin_conv()`**: GINEConv with edge_dim=EDGE_DIM (dynamically computed from concat_feats)
- **Loss function**: BCE (binary cross-entropy) on line ~760

### Notebook Analysis
- **`result_analyser.ipynb`**: Loads model_stats pickles, computes metrics (accuracy, MRR), generates plots for paper figures
  - Reports RA@3 (Recall@3) as primary metric
  - Compares EA-SP, EA-MP, REACT, GRU, GINE baselines
- **`evidence_fusion.ipynb`**: Evidence fusion variant; inspect for multi-model aggregation patterns (EF-SP, EF-MP)
- **`node_feat_gen.ipynb`**: Feature engineering for node embeddings (PCA reduction, IGTD image transformation)
- **`edge_feat_gen.ipynb`**: Feature engineering for edge embeddings (action+column one-hot)

## Debugging Tips

1. **Shape Mismatches**: Check that `display_pca_feats_*` has 181 dimensions (line 50)
2. **Missing Data**: Verify all .pickle files in `chunked_sessions/`, `edge/`, `display_feats/` are extracted
3. **CUDA OOM**: Reduce batch_size calculation or switch to CPU
4. **Pickle Protocol**: Always use `HIGHEST_PROTOCOL` for compatibility
5. **Graph Connectivity**: If getting empty graphs, verify `logic_error_displays` isn't too aggressive

## External Dependencies
- **torch_geometric**: PyG conversion `from_networkx()`, DataLoader, layers (GINEConv, GATConv, global_add_pool)
- **networkx**: DAG construction, BFS/DFS traversal
- **scikit-learn**: PCA, train_test_split, metrics
- **optuna**: Hyperparameter optimization (not used in main scripts; see naive_*.py)

## Key Experimental Details
- **Train/Validation/Test**: 3-way split via `test_id` parameter (4 projects, leave-one-out)
- **5 Runs**: Each config trained 5× with different random seeds for statistical significance
- **Metrics**: 
  - Recall@k (RA@k, with k=3 for main results = "RA3")
  - Mean Reciprocal Rank (MRR) for ranking quality
- **Feature Engineering**: 
  - Node features: PCA projection of display embeddings (181 dims after feature selection from raw features)
  - Edge features: Concatenated action+column one-hot encodings (20 dims)
  - Computed offline via `node_feat_gen.ipynb` and `edge_feat_gen.ipynb`
- **Reproducibility**: All random sources seeded (torch, numpy, random, os) for paper reproducibility
- **Hardware**: Default GPU execution via CUDA; can switch to CPU for debugging
````
