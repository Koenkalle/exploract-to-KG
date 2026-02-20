# Wikidata SPARQL Query Sessions Dataset
 
 This dataset is derived from the ExploratoryQueryingSessions repository, which contains
 30 real SPARQL query sessions from Wikidata query logs.

## Dataset Overview

- **Source**: https://github.com/hartig/ExploratoryQueryingSessions
- **Type**: Graph query exploration sessions
- **Queries**: SPARQL queries on Wikidata knowledge graph
- **Sessions**: 30 real exploratory querying sessions
- **Projects**: 4 (for leave-one-out cross-validation)

## Dataset Structure

### session_repositories/
- `actions.tsv`: Query operations (Filter, Join, Projection, Extension, etc.)
- `displays.tsv`: Simulated query result states

### Key Differences from REACT-IDA Benchmark

This dataset is **distinct** from the original REACT-IDA (cybersecurity log analysis) dataset:
- **Domain**: Wikidata graph queries vs. network packet analysis
- **Operations**: SPARQL graph operations vs. data filter/grouping operations
- **Feature extraction**: Based on SPARQL query structure instead of packet logs
- **Display IDs**: New numbering (starts from 1, not reusing REACT-IDA IDs)

### Feature Files

- `display_feats/display_pca_feats_9999.pickle`: 181-dim random embeddings per display
- `edge/act_feats.pickle`: action type one-hot encodings (number of classes depends on extracted action types)
- `edge/col_action.pickle`: predicate/column one-hot encodings (number of predicates depends on dataset)
- `edge/cond_action.pickle`: condition/operator one-hot encodings (number of operators depends on dataset)

### Important Configuration

When running ExplorAct models with this dataset, always pass:
```python
logic_error_displays=[]  # Empty list - no error display filtering
```

This disables the hardcoded REACT-IDA benchmark error list, which should only apply
to the original cybersecurity dataset.

## Creating the Wikidata Dataset

To generate or regenerate this dataset from the ExploratoryQueryingSessions repository:

```bash
# From the exploract root directory:
python create_wikidata_dataset.py --output wikidata_dataset

# With live SPARQL query execution (fetches actual results from Wikidata):
python create_wikidata_dataset.py --output wikidata_dataset --live

# Use only cached query results (no network calls):
python create_wikidata_dataset.py --output wikidata_dataset --live --cache-only
```

This will:
1. Clone the ExploratoryQueryingSessions repository (if not already present)
2. Parse SPARQL queries from the 30 sessions
3. Create `session_repositories/` with actions.tsv and displays.tsv
4. Generate feature files in `display_feats/` and `edge/`
5. Create train/test splits in `chunked_sessions/`

See `FEATURE_EXTRACTION.md` for details on how node features are extracted from query results.

## Running ExplorAct on Wikidata 

```bash
# From this dataset directory:
python ../ea_sp.py act 20250212 5 0
python ../ea_mp.py col 20250212 5 0
python ../react.py 20250212 5 0
```

## Citation

Original dataset:
Marcelo Arenas et al., "Exploring Exploratory Querying", VLDB 2025.
https://www.vldb.org/pvldb/vol18/p5731-konstantinidis.pdf

WIkidata SPARQL Logs:
Stanislav Malyshev et al., "Getting the Most out of Wikidata", ISWC 2018.
