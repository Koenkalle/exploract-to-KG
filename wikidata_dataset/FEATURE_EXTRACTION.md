# Wikidata Dataset - Feature Extraction Process

## Overview

The node features (181-dimensional embeddings) are extracted from **actual SPARQL query execution results** against the Wikidata endpoint. This follows the ExplorAct paper's methodology of encoding result set characteristics rather than just query structure.

## Feature Extraction Pipeline

### 1. Live Query Execution

For each query in a session, we:

1. **Execute the query** against the Wikidata SPARQL endpoint (`https://query.wikidata.org/sparql`)
2. **Cache the results** in `raw_query_results/display_{id}.json` for reproducibility
3. **Extract features** from the actual result bindings

```python
# Run with live query execution:
python create_wikidata_dataset.py --output wikidata_dataset --live

# Use only cached results (no network calls):
python create_wikidata_dataset.py --output wikidata_dataset --live --cache-only
```

### 2. Result Caching

Query results are cached to disk to avoid repeated API calls:

```
raw_query_results/
├── display_1.json     # SPARQL JSON result format
├── display_2.json
├── display_10.json
└── ...
```

Each cached file contains the standard SPARQL JSON result format:
```json
{
  "head": {"vars": ["var3", "var4", "var6", "var8"]},
  "results": {
    "bindings": [
      {
        "var3": {"type": "literal", "value": "PeerTube", "xml:lang": "af"},
        "var4": {"type": "uri", "value": "http://www.wikidata.org/entity/Q50938515"},
        "var6": {"type": "literal", "value": "Linux"},
        "var8": {"type": "literal", "value": "GNU Affero General Public License..."}
      },
      ...
    ]
  }
}
```

### 3. Feature Extraction from Results

For each result variable in the query response, we compute summary features:

#### Numeric Variables
When a variable contains mostly numeric values, we compute statistical summaries:
- **min, max, mean, std, median, 75th percentile**
- Values are log-transformed to handle large ranges
- Produces 6 features per numeric variable (padded to 8)

Example: `?population` → [log(min), log(max), log(mean), log(std), log(median), log(p75), 0, 0]

#### Categorical/Text Variables
When a variable contains non-numeric values, we create a frequency histogram:
- Values are hashed into 10 bins
- Frequencies are normalized to sum to 1.0
- Captures the distribution of unique values

Example: `?label` with many labels → 10-bin normalized histogram

### 4. Raw Feature Vector Construction

The extraction produces a variable-length raw feature vector:
```python
# Per-variable features concatenated
raw_vec = [
    var1_feats,  # 8 or 10 dims depending on type
    var2_feats,
    var3_feats,
    ...
]
```

### 5. PCA Reduction to 181 Dimensions

After all displays are processed:

1. **Pad all vectors** to the same length (max across all displays)
2. **Standardize** using `StandardScaler` (zero mean, unit variance)
3. **Apply PCA** to reduce to 181 dimensions (or fewer if n_samples < 181)
4. **Pad output** to exactly 181 dimensions if needed

```python
scaler = StandardScaler()
mat_scaled = scaler.fit_transform(mat)
pca = PCA(n_components=min(181, n_samples, n_features))
mat_pca = pca.fit_transform(mat_scaled)
```

### 6. Fallback: Query Structure Features

When live execution fails or returns empty results, we fall back to **structure-based features** extracted from the query text:

| Group | Description | Extraction Method |
|-------|-------------|-------------------|
| 1. Complexity | Query structural complexity | `patterns*10 + filters*5 + optionals*3 + unions*5` |
| 2. Cardinality | Estimated result size | Log-scale bins from LIMIT clause or class count |
| 3. Predicates | Property diversity | Hash P-numbers to 50 bins |
| 4. Classes | Entity class distribution | Hash Q-numbers to 50 bins |
| 5. Filters | Constraint complexity | Character count in FILTER clauses |
| 6. Optionals | Result structure complexity | Variable + OPTIONAL + UNION counts |
| 7. Temporal | Time-related features | Presence of P580/P582/P585/P813 |

The status of each display (live vs fallback) is recorded in `display_feats/raw_query_status.json`.

## Example: Feature Extraction from Live Results

### Query (Display 10)
```sparql
SELECT ?var3 ?var4 ?var6 ?var8 WHERE {
  ?var4 wdt:P31 wd:Q7397 ;
        rdfs:label ?var3 ;
        wdt:P306 ?var5 ;
        wdt:P275 ?var7 .
  ?var5 rdfs:label ?var6 .
  ?var7 rdfs:label ?var8 .
  FILTER(LANG(?var3) != LANG(?var6))
}
LIMIT 5000
```

### Cached Result (truncated)
```json
{
  "head": {"vars": ["var3", "var4", "var6", "var8"]},
  "results": {
    "bindings": [
      {
        "var3": {"xml:lang": "af", "type": "literal", "value": "PeerTube"},
        "var4": {"type": "uri", "value": "http://www.wikidata.org/entity/Q50938515"},
        "var6": {"type": "literal", "value": "Linux"},
        "var8": {"type": "literal", "value": "GNU Affero General Public License..."}
      },
      ...
    ]
  }
}
```

### Extracted Features
For each variable in the result:

| Variable | Type | Feature Extraction |
|----------|------|-------------------|
| `var3` | Text/Literal | 10-bin hash histogram of label values |
| `var4` | URI | 10-bin hash histogram of entity URIs |
| `var6` | Text/Literal | 10-bin hash histogram of OS names |
| `var8` | Text/Literal | 10-bin hash histogram of license names |

Raw vector: ~40 dimensions (10 bins × 4 variables)
After PCA: 181 dimensions

TODO: PCA legacy dims 

## Relationship to ExplorAct Paper

The ExplorAct paper (Section 3.2) describes node feature extraction:

> "For each column in an action result, generate probability vectors based on column type"

For Wikidata, we implement this by **executing queries and analyzing actual results**:

| Paper Concept | Wikidata Implementation |
|---------------|------------------------|
| Numerical columns | Statistical summaries (min, max, mean, std, median, p75) with log-transform |
| Categorical columns | 10-bin hash histograms of unique values |
| Text columns | 10-bin hash histograms of string values |
| PCA reduction | StandardScaler + PCA to 181 dimensions |

TODO: PCA legacy dims 

### Key Difference from Query-Structure-Only Approach

The previous approach only analyzed SPARQL query **structure** (predicates, filters, etc.). The current approach:

1. **Executes queries** against Wikidata endpoint with caching
2. **Analyzes actual result values** (not just query structure)
3. **Computes real statistics** from returned bindings
4. **Falls back to structure-based features** only when execution fails


## Feature Verification

To verify features are properly extracted from live results:

```python
import pickle
import json
import numpy as np

# Check extraction status
with open('display_feats/raw_query_status.json', 'r') as f:
    status = json.load(f)

live_count = sum(1 for v in status.values() if v.get('status') == 'live_fetched')
fallback_count = sum(1 for v in status.values() if v.get('status') == 'fallback_simulated')
print(f"Live fetched: {live_count}, Fallback: {fallback_count}")

# Load final PCA features
with open('display_feats/display_pca_feats_9999.pickle', 'rb') as f:
    feats = pickle.load(f)

# Check diversity
print(f"Displays: {len(feats)}")
all_feats = np.array([feats[k] for k in feats])
print(f"Feature shape: {all_feats.shape}")

# Compute pairwise distances
from scipy.spatial.distance import pdist
distances = pdist(all_feats, metric='euclidean')
print(f"Distance range: {np.min(distances):.4f} - {np.max(distances):.4f}")

# Check raw features before PCA
with open('display_feats/raw_display_feats.pickle', 'rb') as f:
    raw_feats = pickle.load(f)
raw_dims = [v.shape[0] for v in raw_feats.values()]
print(f"Raw feature dims: min={min(raw_dims)}, max={max(raw_dims)}")
```

Expected results:
- Most displays should be `live_fetched` (not `fallback_simulated`)
- 178 displays total
- Final shape: (178, 181)
- L2 distances between different displays: 0.3-1.4
- Raw feature dimensions vary by number of result variables

## Output Files

| File | Description |
|------|-------------|
| `raw_query_results/display_*.json` | Cached SPARQL JSON responses |
| `display_feats/raw_display_feats.pickle` | Raw feature vectors before PCA |
| `display_feats/raw_query_status.json` | Status of each display (live vs fallback) |
| `display_feats/display_pca_feats_9999.pickle` | Final 181-dim PCA-reduced features |
| `display_feats/display_queries.json` | Display ID → original SPARQL query mapping |

TODO: PCA legacy dims and naming

## Future Improvements

1. **Richer Result Analysis**
   - Type detection for literals (dates, coordinates, quantities)
   - Language distribution analysis for multilingual labels
   - URI structure analysis for entity/property references

2. **Semantic Embeddings**
   - Use Wikidata entity embeddings (PyKEEN, TransE, etc.)
   - Compute average entity embedding per result set
   - Leverage property semantics from Wikidata ontology

3. **Temporal Analysis**
   - Extract and bin date/time values
   - Compute time range features
   - Analyze temporal predicates (P580, P582, P585)

## References

**ExplorAct Paper (Section 3.2):**
- "For each column in an action result, generate probability vectors based on column type"
- Node features: PCA reduction to 181 dimensions
- Edge features: One-hot action type + predicate

**Wikidata Documentation:**
- https://www.wikidata.org/
- https://query.wikidata.org/ (SPARQL endpoint)
- https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service/queries

**SPARQL Specification:**
- https://www.w3.org/TR/sparql11-query/
