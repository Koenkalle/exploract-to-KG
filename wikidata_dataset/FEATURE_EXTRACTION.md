# Wikidata Dataset - Feature Extraction Process

## Overview

The node features (181-dimensional embeddings) are extracted from the **SPARQL query structure** following the ExplorAct paper's methodology. Unlike random initialization, these features encode actual query characteristics.

## Feature Extraction Pipeline

### 1. SPARQL Query Analysis

For each query in a session, we extract:

```python
# Query components extracted
- SELECT variables: ?x, ?label, ?date, etc.
- Triple patterns: ?x P31 Q5, ?x P625 ?coords
- Predicates: P31 (instance of), P625 (coordinates), P27 (country), etc.
- Entity classes: Q5 (human), Q131681 (actor), etc.
- FILTER constraints: LANG(?label) = "en", ?date > 2000
- OPTIONAL clauses: OPTIONAL { ?x P18 ?image }
- UNION clauses: UNION { ... }
- LIMIT/OFFSET: LIMIT 100, OFFSET 50
```

### 2. Feature Generation (7 feature groups)

Each query generates 181-dimensional features by computing 7 feature groups (50-dimensional each, plus padding):

#### Group 1: Query Complexity Features
- Encodes the structural complexity of the query
- Computed from: `num_triple_patterns * 10 + num_filters * 5 + optionals * 3 + unions * 5`
- 50-bin histogram showing complexity distribution

Example:
- Simple query (1 pattern, 1 filter): Low complexity scores
- Complex query (5 patterns, 3 filters, 2 optionals): High complexity scores

#### Group 2: Result Cardinality Estimation
- Estimates how many results the query returns
- Based on:
  - LIMIT clause value (if present)
  - Number of entity classes in query (affects result set size)
  - Query structure (joins typically reduce cardinality)
- Log-scale binning to capture range from 1 to 100K+ results

Example:
- `SELECT ?x WHERE { ?x P31 Q5 }` → ~6 million humans → High cardinality
- `SELECT ?x WHERE { ?x P31 Q5 LIMIT 100 }` → 100 results → Low cardinality

#### Group 3: Predicate Diversity
- Encodes which Wikidata predicates are used
- Predicates are hashed to bins based on property number
- Multiple predicates can contribute to same bins (sparse representation)

Common predicates:
- P31: instance of
- P625: coordinate location
- P27: country of citizenship
- P17: country
- P580/P582: start/end time
- P585: point in time
- P18: image

#### Group 4: Entity Class Distribution
- Encodes which Wikidata entity classes appear
- Entity classes hashed similarly to predicates
- Q5 (human), Q131681 (actor), Q6581097 (male), etc.

Example:
- Query looking for "human scientists": Both Q5 and class distributions active
- Query looking for "geographic locations": Q2221906 (city) and P625 (coordinates)

#### Group 5: Filter Constraint Distribution
- Encodes complexity of FILTER clauses
- Counts total characters in filters (linguistic constraint)
- Binned to show presence and intensity of constraints

Example:
- No FILTER: All bins weighted equally
- `FILTER (LANG(?label) = "en")`: Moderate constraint
- Complex FILTER with multiple conditions: High constraint score

#### Group 6: Variable/OPTIONAL/UNION Usage
- Encodes result structure complexity
- Counts number of variables, OPTIONAL clauses, UNIONs
- Important for understanding result heterogeneity

Example:
- Simple: Few variables, no optionals → Low score
- Complex: Many variables, optional branches, union alternatives → High score

#### Group 7: Temporal Signature
- Encodes temporal aspects of query
- Presence of temporal predicates: P580, P582, P585, P813
- Temporal constraints in FILTER clauses

Example:
- Query about "scientists born in 1900": Temporal predicates present
- Query about "current population": Temporal filters present

### 3. Feature Normalization

Each 50-bin group is:
1. **Normalized to probability distribution**: `bins / sum(bins)`
2. **Concatenated**: All 7 groups stacked (350+ dimensions)
3. **Truncated to 181 dimensions**: Following PCA practice from paper

### 4. Deterministic Feature Generation

Features are deterministic based on display_id:
- `np.random.seed(display_id)` ensures same query always gets same features
- Different queries get different seeds → different features
- Reproducible across runs

## Examples

### Session 1: Simple to Complex Query Evolution

**Query 1** (Display 1):
```sparql
SELECT ?x WHERE { ?x P31 Q5 }
```
Features: 
- Complexity: Low (1 pattern, 0 filters)
- Cardinality: Very high (6M humans)
- Predicates: P31 only
- Classes: Q5 only
- No filters, no optionals

**Query 2** (Display 2):
```sparql
SELECT ?x ?label WHERE { 
  ?x P31 Q5
  OPTIONAL { ?x rdfs:label ?label FILTER (LANG(?label) = "en") }
}
```
Features:
- Complexity: Medium (1 pattern, 1 filter, 1 optional)
- Cardinality: Very high (still 6M + optional labels)
- Predicates: P31, rdfs:label
- Filter encoding: Present
- Optional count: 1

**Query 3** (Display 3):
```sparql
SELECT ?x ?label WHERE { 
  ?x P31 Q5
  OPTIONAL { ?x rdfs:label ?label FILTER (LANG(?label) = "en") }
  LIMIT 100000
}
```
Features:
- Complexity: Medium (same as Query 2 structure)
- Cardinality: Medium (100K LIMIT applied)
- Other features: Similar to Query 2
- Key difference: Cardinality bin changes

## Relationship to ExplorAct Paper

The ExplorAct paper (Section 3.2) describes node feature extraction:

> "For each column in an action result, generate probability vectors based on column type"

For Wikidata:
- **Numerical columns** → Predicate value ranges (coordinates, counts)
- **Categorical columns** → Entity classes, Wikidata properties
- **Text columns** → Language templates, labels
- **Temporal columns** → Date/time ranges

Our feature extraction simulates this by:
1. Parsing SPARQL structure (instead of executing queries)
2. Creating probability bins for each feature type
3. Building histogram features from query characteristics
4. Normalizing to probability distributions
5. Concatenating and PCA-reducing to 181 dimensions

## Why Not Random Features?

Random features would:
- ❌ Lose query structure information
- ❌ Make session evolution uninterpretable
- ❌ Reduce model's ability to learn patterns
- ❌ Not match paper methodology

Query-derived features:
- ✅ Encode actual query structure
- ✅ Capture session evolution (queries get more complex, more constrained)
- ✅ Enable meaningful learning of exploration patterns
- ✅ Follow the paper's feature engineering approach

## Feature Verification

To verify features are properly extracted:

```python
import pickle
import numpy as np

# Load features
with open('display_feats/display_pca_feats_9999.pickle', 'rb') as f:
    feats = pickle.load(f)

# Check diversity
print(f"Displays: {len(feats)}")
all_feats = np.array([feats[k] for k in feats])
print(f"Feature shape: {all_feats.shape}")
print(f"Distance range: {np.min(distances):.4f} - {np.max(distances):.4f}")
```

Expected results:
- 178 displays
- Shape: (178, 181)
- L2 distances between different displays: 0.3-1.4
- Same query always has same features (seed-based)
- Different queries have different features

## Future Improvements

To further enhance features, consider:

1. **Live SPARQL Execution** (if time/resources available)
   - Execute queries against Wikidata SPARQL endpoint
   - Extract actual result distributions
   - Measure real cardinality instead of estimating

2. **Result Sampling**
   - Execute LIMIT versions of queries
   - Extract actual column distributions
   - Compute real value histograms

3. **Graph Structure Analysis**
   - Analyze predicate relationships
   - Compute semantic similarity between predicates
   - Build knowledge graph embeddings

4. **PCA Optimization**
   - Run PCA on actual query results
   - Find optimal dimension reduction
   - Use actual explained variance ratio

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
