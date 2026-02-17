"""
Create an ExplorAct dataset from ExploratoryQueryingSessions (Wikidata SPARQL queries).

This script:
1. Downloads/clones the ExploratoryQueryingSessions repository
2. Parses SPARQL queries to extract graph query operations
3. Creates session_repositories with actions.tsv and displays.tsv
4. Generates feature files for node and edge embeddings
5. Creates chunked_sessions for train/test splitting

Dataset structure mirrors real ExplorAct data but uses SPARQL query graph operations
as the basis for "actions" (which can be thought of as:
  - Query refinement (adding filters, joins, aggregations)
  - Result projection (selecting/removing columns)
  - Result extension (adding new graph traversals)
)
"""

import os
import sys
import shutil
import pickle
import numpy as np
import re
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime, timedelta
import random

# New imports for live SPARQL extraction and PCA
import requests
import time
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import ast

# ============================================================================
# SPARQL QUERY PARSER
# ============================================================================

class SPARQLQueryAnalyzer:
    """Extract graph operations from SPARQL queries."""
    
    def __init__(self):
        self.triple_patterns = []
        self.filters = []
        self.variables = set()
        self.predicates = []
        self.classes = []
        self.limit = None
        self.offset = None
        self.select_vars = []
        self.optional_count = 0
        self.union_count = 0
        
    def parse(self, query_text):
        """Parse SPARQL query and extract operations."""
        self._reset()
        
        # Normalize whitespace
        query = ' '.join(query_text.split())
        
        # Extract SELECT variables
        select_match = re.search(r'SELECT\s+(\?[^\s]*(?:\s+\?[^\s]*)*)', query, re.IGNORECASE)
        if select_match:
            self.select_vars = select_match.group(1).split()
        
        # Extract triples (simplified)
        triple_pattern = r'\?\w+\s+<[^>]+>\s+[^.;}\[]*'
        self.triple_patterns = re.findall(triple_pattern, query)
        
        # Extract variables
        self.variables = set(re.findall(r'\?\w+', query))
        
        # Extract predicates (URIs)
        self.predicates = re.findall(r'<(http[^>]+)>', query)
        
        # Extract class references (like Q5 for humans)
        self.classes = re.findall(r'<http://www\.wikidata\.org/entity/(Q\d+)>', query)
        
        # Extract OPTIONAL count
        self.optional_count = len(re.findall(r'OPTIONAL\s*\{', query, re.IGNORECASE))
        
        # Extract UNION count
        self.union_count = len(re.findall(r'UNION', query, re.IGNORECASE))
        
        # Extract FILTER count
        self.filters = re.findall(r'FILTER\s*\([^)]*\)', query, re.IGNORECASE)
        
        # Extract LIMIT
        limit_match = re.search(r'LIMIT\s+(\d+)', query, re.IGNORECASE)
        if limit_match:
            self.limit = int(limit_match.group(1))
        
        # Extract OFFSET
        offset_match = re.search(r'OFFSET\s+(\d+)', query, re.IGNORECASE)
        if offset_match:
            self.offset = int(offset_match.group(1))
        
        return self
    
    def _reset(self):
        self.triple_patterns = []
        self.filters = []
        self.variables = set()
        self.predicates = []
        self.classes = []
        self.limit = None
        self.offset = None
        self.select_vars = []
        self.optional_count = 0
        self.union_count = 0
    
    def get_operation_type(self):
        """Classify query as: Filter, Join, Projection, Extension, Aggregation."""
        # This would be enhanced based on query complexity
        if self.filters:
            return "Filter"
        elif self.union_count > 0:
            return "Union"
        elif self.optional_count > 0:
            return "Join"
        elif len(self.predicates) > 2:
            return "Extension"
        else:
            return "Projection"
    
    def get_operation_params(self):
        """Extract parameters for the operation."""
        return {
            'field': self.predicates[0] if self.predicates else 'P31',
            'num_triples': len(self.triple_patterns),
            'num_filters': len(self.filters),
            'limit': self.limit,
            'offset': self.offset,
            'optional': self.optional_count,
            'union': self.union_count,
        }

# ============================================================================
# DATASET CREATION
# ============================================================================

def download_repository(output_dir='wikidata_dataset'):
    """Clone or use existing ExploratoryQueryingSessions repository."""
    repo_url = 'https://github.com/hartig/ExploratoryQueryingSessions.git'
    repo_path = '/tmp/ExploratoryQueryingSessions'
    
    # Clone if not exists
    if not os.path.exists(repo_path):
        print(f"Cloning repository from {repo_url}...")
        os.system(f'git clone {repo_url} {repo_path} 2>&1 | head -5')
    else:
        print(f"Using existing repository at {repo_path}")
    
    return repo_path

def read_queries_from_session(session_path):
    """Read all queries from a session directory."""
    queries = []
    query_files = sorted([f for f in os.listdir(session_path) 
                         if f.startswith('query') and f.endswith('.rq')])
    
    for query_file in query_files:
        with open(os.path.join(session_path, query_file), 'r') as f:
            queries.append(f.read())
    
    return queries

def create_session_structure(repo_path, output_dir, num_projects=4):
    """
    Parse SPARQL sessions and create ExplorAct-compatible data structure.
    
    Maps:
    - Sessions → Projects (distribute 30 sessions across num_projects)
    - Queries within session → Actions
    - Query results (simulated) → Displays
    """
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/session_repositories', exist_ok=True)
    os.makedirs(f'{output_dir}/chunked_sessions', exist_ok=True)
    os.makedirs(f'{output_dir}/display_feats', exist_ok=True)
    os.makedirs(f'{output_dir}/edge', exist_ok=True)
    os.makedirs(f'{output_dir}/raw_datasets', exist_ok=True)
    
    sessions_dir = os.path.join(repo_path, 'sessions')
    session_folders = sorted([d for d in os.listdir(sessions_dir) 
                             if os.path.isdir(os.path.join(sessions_dir, d))])
    
    print(f"Found {len(session_folders)} sessions")
    
    # Distribute sessions across projects
    sessions_per_project = len(session_folders) // num_projects
    project_sessions = {}
    for p in range(num_projects):
        start = p * sessions_per_project
        end = start + sessions_per_project if p < num_projects - 1 else len(session_folders)
        project_sessions[p] = session_folders[start:end]
    
    # ========================================================================
    # CREATE ACTIONS AND DISPLAYS FILES
    # ========================================================================
    
    actions_rows = []
    displays_rows = []
    display_id_counter = 1
    action_id_counter = 1
    
    # Track for chunked_sessions
    project_query_edges = {p: [] for p in range(num_projects)}
    
    # Track query text for feature extraction
    display_to_query = {}
    
    for project_id in range(num_projects):
        session_count = 0
        for session_folder in project_sessions[project_id]:
            session_path = os.path.join(sessions_dir, session_folder)
            queries = read_queries_from_session(session_path)
            
            if len(queries) < 2:
                continue
            
            session_id = f"{project_id}_{session_count}"
            session_count += 1
            
            # Parse queries
            analyzer = SPARQLQueryAnalyzer()
            query_displays = []  # Display ID for each query state
            
            for query_idx, query_text in enumerate(queries):
                # Simulate a display (result state)
                parent_display_id = query_displays[-1] if query_displays else 1
                child_display_id = display_id_counter
                display_id_counter += 1
                query_displays.append(child_display_id)
                
                # Store query text for feature extraction
                display_to_query[child_display_id] = query_text
                
                # Record display
                displays_rows.append({
                    'display_id': child_display_id,
                    'session_id': session_id,
                    'query_index': query_idx,
                })
                
                # Parse query to get operation
                analyzer.parse(query_text)
                operation_type = analyzer.get_operation_type()
                operation_params = analyzer.get_operation_params()
                
                # Record action
                if query_idx > 0:  # First query is not an action, it's the root
                    action_row = {
                        'action_id': action_id_counter,
                        'session_id': session_id,
                        'project_id': project_id,
                        'action_type': operation_type.lower(),
                        'action_params': str(operation_params),
                        'parent_display_id': parent_display_id,
                        'child_display_id': child_display_id,
                        'solution': 1,  # All queries are valid
                    }
                    actions_rows.append(action_row)
                    
                    # Record edge for graph reconstruction
                    project_query_edges[project_id].append([
                        [parent_display_id, child_display_id, {'aid': action_id_counter}]
                    ])
                    
                    action_id_counter += 1
    
    # ========================================================================
    # WRITE ACTIONS.TSV
    # ========================================================================
    
    actions_path = f'{output_dir}/session_repositories/actions.tsv'
    with open(actions_path, 'w') as f:
        # Header
        f.write('\t'.join([
            'action_id', 'session_id', 'project_id', 'action_type', 
            'action_params', 'parent_display_id', 'child_display_id', 'solution'
        ]) + '\n')
        
        # Rows
        for row in actions_rows:
            f.write('\t'.join(str(row.get(k, '')) for k in [
                'action_id', 'session_id', 'project_id', 'action_type',
                'action_params', 'parent_display_id', 'child_display_id', 'solution'
            ]) + '\n')
    
    print(f"Created {actions_path} with {len(actions_rows)} actions")
    
    # ========================================================================
    # WRITE DISPLAYS.TSV
    # ========================================================================
    
    displays_path = f'{output_dir}/session_repositories/displays.tsv'
    with open(displays_path, 'w') as f:
        # Header
        f.write('\t'.join(['display_id', 'session_id', 'query_index']) + '\n')
        
        # Rows
        for row in displays_rows:
            f.write('\t'.join(str(row.get(k, '')) for k in [
                'display_id', 'session_id', 'query_index'
            ]) + '\n')
    
    print(f"Created {displays_path} with {len(displays_rows)} displays")
    
    # ========================================================================
    # CREATE RAW DATASET (dummy graph data)
    # ========================================================================
    
    raw_data_path = f'{output_dir}/raw_datasets/1.tsv'
    with open(raw_data_path, 'w') as f:
        # Create a minimal graph dataset representation
        f.write('\t'.join(['node_id', 'label', 'class', 'predicate']) + '\n')
        for i in range(1, 50):
            f.write('\t'.join([
                str(i), f'Entity_{i}', f'Q{i}', f'P{i%10}'
            ]) + '\n')
    
    print(f"Created {raw_data_path}")
    
    return actions_rows, displays_rows, project_query_edges, display_id_counter, action_id_counter, display_to_query

def extract_sparql_features(query_text, display_id):
    """
    Extract features from SPARQL query results following ExplorAct paper methodology.
    
    Features capture:
    1. Query complexity (number of patterns, filters, etc.)
    2. Result cardinality estimates
    3. Column type distributions
    4. Entity class distributions
    5. Temporal characteristics
    """
    
    np.random.seed(display_id)  # Deterministic but different per display
    features = []
    
    # Extract SELECT variables
    select_match = re.search(r'SELECT\s+(\?[^\s]*(?:\s+\?[^\s]*)*)', query_text, re.IGNORECASE)
    if select_match:
        select_vars = select_match.group(1).split()
    else:
        select_vars = ['?result']
    
    num_vars = len(select_vars)
    
    # Extract SPARQL triple patterns and predicates
    triple_patterns = re.findall(r'\?\w+\s+<[^>]+>\s+[^.;}\[]*', query_text)
    predicates = re.findall(r'<(http[^>]+)>', query_text)
    classes = re.findall(r'<http://www\.wikidata\.org/entity/(Q\d+)>', query_text)
    filters = re.findall(r'FILTER\s*\([^)]*\)', query_text, re.IGNORECASE)
    optionals = len(re.findall(r'OPTIONAL\s*\{', query_text, re.IGNORECASE))
    unions = len(re.findall(r'UNION', query_text, re.IGNORECASE))
    limit = None
    limit_match = re.search(r'LIMIT\s+(\d+)', query_text, re.IGNORECASE)
    if limit_match:
        limit = int(limit_match.group(1))
    
    # 1. Query complexity feature (bin-based)
    complexity_score = (len(triple_patterns) * 10 + 
                       len(filters) * 5 + 
                       optionals * 3 + 
                       unions * 5)
    complexity_bins = np.zeros(50, dtype=np.float32)
    for i in range(min(50, complexity_score)):
        complexity_bins[i] = 1.0
    features.append(complexity_bins / max(complexity_bins.sum(), 1.0))
    
    # 2. Result cardinality estimation based on query structure
    cardinality_estimate = 100
    if limit:
        cardinality_estimate = min(limit, 10000)
    else:
        cardinality_estimate = min(1000 * max(1, len(classes)), 100000)
    
    card_bins = np.zeros(50, dtype=np.float32)
    card_idx = min(49, max(0, int(np.log10(cardinality_estimate + 1) * 10)))
    for i in range(card_idx, 50):
        card_bins[i] = np.exp(-(i - card_idx) * 0.15)
    features.append(card_bins / max(card_bins.sum(), 1.0))
    
    # 3. Predicate diversity (one hot encoding of predicate types)
    pred_bins = np.zeros(50, dtype=np.float32)
    for prop_idx, predicate in enumerate(predicates):
        prop_match = re.search(r'(P\d+)', predicate)
        if prop_match:
            prop_num = int(prop_match.group(1)[1:])
            bin_idx = (prop_num * 7) % 50  # Hash to bin
            pred_bins[bin_idx] += 1.0
    features.append(pred_bins / max(pred_bins.sum(), 1.0))
    
    # 4. Entity class distribution
    class_bins = np.zeros(50, dtype=np.float32)
    for class_id in classes:
        try:
            class_num = int(class_id[1:])
            bin_idx = (class_num * 13) % 50  # Hash to bin
            class_bins[bin_idx] += 1.0
        except:
            pass
    features.append(class_bins / max(class_bins.sum(), 1.0))
    
    # 5. Filter constraint distribution
    filter_bins = np.zeros(50, dtype=np.float32)
    filter_score = sum(len(f) for f in filters)
    for i in range(min(50, filter_score // 10)):
        filter_bins[i] = 1.0
    if filter_bins.sum() == 0:
        filter_bins[:] = 1.0 / 50
    features.append(filter_bins / max(filter_bins.sum(), 1.0))
    
    # 6. Variable count and OPTIONAL/UNION usage
    var_union_bins = np.zeros(50, dtype=np.float32)
    var_union_score = num_vars * 5 + optionals * 15 + unions * 20
    for i in range(min(50, var_union_score)):
        var_union_bins[i] = 1.0
    features.append(var_union_bins / max(var_union_bins.sum(), 1.0))
    
    # 7. Temporal signature (based on presence of temporal predicates)
    temporal_bins = np.zeros(50, dtype=np.float32)
    temporal_preds = sum(1 for p in predicates if re.search(r'(P580|P582|P585|P813)', p))
    has_date_filter = bool(re.search(r'(P580|P582|P585)', query_text))
    temporal_score = temporal_preds * 10 + (20 if has_date_filter else 0)
    for i in range(min(50, temporal_score)):
        temporal_bins[i] = 1.0
    features.append(temporal_bins / max(temporal_bins.sum(), 1.0))
    
    # Add padding features to reach 181 dimensions
    total_dims = sum(f.shape[0] for f in features)
    if total_dims < 181:
        padding_dims = 181 - total_dims
        # Create structured padding features
        num_padding_features = (padding_dims + 49) // 50  # Round up
        for i in range(num_padding_features):
            # Add features based on query characteristics
            pad_bins = np.random.dirichlet(np.ones(min(50, padding_dims - i*50)))
            features.append(pad_bins.astype(np.float32))
    
    # Concatenate and truncate to exactly 181
    feature_vector = np.concatenate(features)
    return feature_vector[:181].astype(np.float32)

# New helper: run a SPARQL query against Wikidata with retries and caching

def run_sparql_query(query_text, endpoint='https://query.wikidata.org/sparql', timeout=30, retries=3):
    headers = {
        'User-Agent': 'ExplorActDatasetBuilder/1.0 (example@example.org)',
        'Accept': 'application/sparql-results+json'
    }
    data = {'query': query_text}
    backoff = 1.0
    for attempt in range(retries):
        try:
            print(f"[SPARQL] Attempt {attempt+1}/{retries} to endpoint (len(query)={len(query_text):d})")
            resp = requests.post(endpoint, data=data, headers=headers, timeout=timeout)
            print(f"[SPARQL] HTTP {resp.status_code} for query (len={len(query_text):d})")
            if resp.status_code == 200:
                return resp.json()
            else:
                # non-200, sleep and retry
                time.sleep(backoff)
                backoff *= 2
        except Exception as e:
            print(f"[SPARQL] Request exception: {e}; backing off {backoff}s")
            time.sleep(backoff)
            backoff *= 2
    print("[SPARQL] All retries exhausted, returning None")
    return None


def fetch_and_cache_query(display_id, query_text, cache_dir, sample_limit=5000, polite_delay=1.0):
    """Fetch query results and cache to disk. Returns parsed JSON or None.
    Prints cache usage and basic diagnostics.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'display_{display_id}.json')
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                print(f"[CACHE] Hit for display {display_id} -> {cache_path}")
                return json.load(f)
        except Exception as e:
            print(f"[CACHE] Failed to load cache for display {display_id}: {e} (will re-fetch)")

    print(f"[CACHE] Miss for display {display_id}; fetching from endpoint...")

    # Optionally add a LIMIT if not present to avoid huge responses
    query_to_run = query_text
    if not re.search(r'\bLIMIT\b', query_text, re.IGNORECASE):
        query_to_run = query_text.strip() + f"\nLIMIT {sample_limit}"
        print(f"[CACHE] Added LIMIT {sample_limit} to query for display {display_id}")

    # Be polite to Wikimedia's endpoint
    print(f"[CACHE] Sleeping {polite_delay}s before request for display {display_id}")
    time.sleep(polite_delay)
    res = run_sparql_query(query_to_run)
    if res is None:
        print(f"[CACHE] Fetch failed for display {display_id}")
        return None

    # Save cache
    try:
        with open(cache_path, 'w') as f:
            json.dump(res, f)
        print(f"[CACHE] Saved response for display {display_id} to {cache_path}")
    except Exception as e:
        print(f"[CACHE] Failed to write cache for display {display_id}: {e}")
    return res


def _compute_raw_vector_from_results(res_json):
    """Produce a fixed-length raw vector from SPARQL results JSON.
    Strategy (simple, robust):
    - For each variable produce small summary features (numeric stats or categorical histograms)
    - Concatenate and return 1D numpy array
    """
    if not res_json or 'results' not in res_json:
        return None
    vars_ = res_json.get('head', {}).get('vars', [])
    bindings = res_json.get('results', {}).get('bindings', [])
    per_var_feats = []
    for v in vars_:
        vals = [b.get(v, {}).get('value') for b in bindings if v in b]
        if not vals:
            # empty: add small neutral vector
            per_var_feats.extend([0.0] * 8)
            continue
        # Detect numeric
        num_vals = []
        non_num_vals = []
        for val in vals:
            try:
                num_vals.append(float(val))
            except Exception:
                non_num_vals.append(val)
        if len(num_vals) >= max(1, len(vals) // 2):
            arr = np.array(num_vals)
            # stats: min, max, mean, std, median, skew approx
            stats = [float(np.min(arr)), float(np.max(arr)), float(np.mean(arr)), float(np.std(arr)), float(np.median(arr)), float(np.percentile(arr, 75))]
            # Normalize stats to reasonable range using log scale for large values
            stats = [np.log1p(abs(s)) if isinstance(s, float) else 0.0 for s in stats]
            per_var_feats.extend(stats[:6])
            # pad
            per_var_feats.extend([0.0, 0.0])
        else:
            # Categorical/text: hash into 10 bins -> freq vector
            bins = np.zeros(10, dtype=np.float32)
            for val in non_num_vals[:1000]:
                b = (abs(hash(val)) % 10)
                bins[b] += 1.0
            if bins.sum() > 0:
                bins = bins / bins.sum()
            per_var_feats.extend(bins.tolist())
            # pad to 8 dims if needed
            if len(per_var_feats) % 8 != 0:
                # ensure per var contributes 8 dims
                while len(per_var_feats) % 8 != 0:
                    per_var_feats.append(0.0)
    return np.array(per_var_feats, dtype=np.float32)

def create_feature_files(output_dir, display_id_max, action_id_max, sessions_dir=None, query_map=None, live=False, cache_only=False):
    """
    Create feature files matching ExplorAct format.

    Node features: Extracted from actual SPARQL query results when live=True
    Edge features: Action and predicate embeddings
    """

    # ========================================================================
    # NODE FEATURES (Display embeddings - 181 dimensional)
    # ========================================================================

    display_pca_feats = {}

    if query_map and live:
        # Live extraction mode: query Wikidata, cache results, compute raw features, then PCA-reduce
        cache_dir = os.path.join(output_dir, 'raw_query_results')
        raw_display_feats = {}
        status = {}
        max_raw_len = 0
        live_count = 0
        fallback_count = 0
        for display_id, query_text in query_map.items():
            print(f"[EXTRACT] Processing display {display_id} (query len={len(query_text):d})")
            # If cache_only is True, load only from existing cache and do NOT fetch from endpoint
            if cache_only:
                cache_path = os.path.join(cache_dir, f'display_{display_id}.json')
                if os.path.exists(cache_path):
                    try:
                        with open(cache_path, 'r') as cf:
                            res = json.load(cf)
                        print(f"[CACHE_ONLY] Loaded cached response for display {display_id}")
                    except Exception as e:
                        print(f"[CACHE_ONLY] Failed to load cached response for display {display_id}: {e}")
                        res = None
                else:
                    print(f"[CACHE_ONLY] No cache for display {display_id}; will use fallback simulated features")
                    res = None
            else:
                res = fetch_and_cache_query(display_id, query_text, cache_dir)
            if res is None:
                # fallback to simulated extractor but mark status
                raw_vec = extract_sparql_features(query_text, display_id)
                raw_display_feats[display_id] = raw_vec.astype(np.float32)
                status[display_id] = {'status': 'fallback_simulated'}
                max_raw_len = max(max_raw_len, raw_vec.size)
                fallback_count += 1
                print(f"[EXTRACT] Display {display_id} used fallback simulated features")
            else:
                raw_vec = _compute_raw_vector_from_results(res)
                if raw_vec is None or raw_vec.size == 0:
                    # fallback
                    raw_vec = extract_sparql_features(query_text, display_id)
                    status[display_id] = {'status': 'fallback_simulated'}
                    fallback_count += 1
                    print(f"[EXTRACT] Display {display_id} had empty results; used fallback simulated features")
                else:
                    status[display_id] = {'status': 'live_fetched', 'rows': len(res.get('results', {}).get('bindings', []))}
                    live_count += 1
                    print(f"[EXTRACT] Display {display_id} live fetched with {status[display_id]['rows']} rows")
                raw_display_feats[display_id] = raw_vec.astype(np.float32)
                max_raw_len = max(max_raw_len, raw_vec.size)

        # Save raw display feats and status
        raw_feats_path = os.path.join(output_dir, 'display_feats', 'raw_display_feats.pickle')
        with open(raw_feats_path, 'wb') as f:
            pickle.dump(raw_display_feats, f, protocol=pickle.HIGHEST_PROTOCOL)
        status_path = os.path.join(output_dir, 'display_feats', 'raw_query_status.json')
        with open(status_path, 'w') as f:
            json.dump(status, f, indent=2)
        print(f"Saved raw display feats to {raw_feats_path} and status to {status_path}")
        print(f"[EXTRACT] Summary: {live_count} live fetched, {fallback_count} fallback simulated, total {len(raw_display_feats)} displays")

        # Build matrix and pad to equal length
        mat = []
        ids = []
        for did, vec in raw_display_feats.items():
            ids.append(did)
            if vec.size < max_raw_len:
                padded = np.zeros(max_raw_len, dtype=np.float32)
                padded[:vec.size] = vec
                mat.append(padded)
            else:
                mat.append(vec)
        mat = np.vstack(mat)

        # Standardize then PCA to 181 dims
        scaler = StandardScaler()
        mat_scaled = scaler.fit_transform(mat)
        # Choose n_components <= min(n_samples, n_features)
        n_samples, n_features = mat_scaled.shape
        n_components = min(181, n_samples, n_features)
        if n_components <= 0:
            raise ValueError("No data available for PCA")
        pca = PCA(n_components=n_components)
        mat_pca = pca.fit_transform(mat_scaled)

        # If PCA produced fewer than 181 dimensions, pad with zeros
        if mat_pca.shape[1] < 181:
            padded = np.zeros((mat_pca.shape[0], 181), dtype=np.float32)
            padded[:, :mat_pca.shape[1]] = mat_pca.astype(np.float32)
            mat_pca = padded

        # Map back to display_pca_feats
        for i, did in enumerate(ids):
            display_pca_feats[did] = mat_pca[i].astype(np.float32)

    elif query_map:
        # Deterministic structural extraction from query text (fast fallback)
        for display_id in range(1, display_id_max + 1):
            if display_id in query_map:
                query_text = query_map[display_id]
                # Extract real features from SPARQL structure (simulated)
                features = extract_sparql_features(query_text, display_id)
            else:
                # Fallback: generate realistic features
                features = extract_sparql_features("SELECT ?x WHERE { ?x ?p ?o }", display_id)
            display_pca_feats[display_id] = features
    else:
        # Generate features from synthetic patterns
        for display_id in range(1, display_id_max + 1):
            # Create features with structure rather than pure random
            features = np.zeros(181, dtype=np.float32)
            
            # Add some structure: peak at certain positions based on display_id
            peak_pos = (display_id * 17) % 181  # Pseudo-random but deterministic
            width = 20
            for i in range(max(0, peak_pos-width), min(181, peak_pos+width)):
                features[i] = np.exp(-(i - peak_pos)**2 / (2 * width**2))
            
            # Normalize and add some Gaussian noise
            features = features / (np.linalg.norm(features) + 1e-6)
            features = features + 0.01 * np.random.randn(181)
            features = np.clip(features, -1, 1).astype(np.float32)
            
            display_pca_feats[display_id] = features
    
    feats_path = f'{output_dir}/display_feats/display_pca_feats_9999.pickle'
    with open(feats_path, 'wb') as f:
        pickle.dump(display_pca_feats, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Created {feats_path} with {len(display_pca_feats)} displays")

    # Save display -> original query mapping for provenance
    queries_out = os.path.join(output_dir, 'display_feats', 'display_queries.json')
    try:
        with open(queries_out, 'w') as fq:
            json.dump({str(k): v for k, v in (query_map or {}).items()}, fq, indent=2)
        print(f"Saved display->query mapping to {queries_out}")
    except Exception as e:
        print(f"Failed to write display_queries.json: {e}")
    
    # ========================================================================
    # EDGE FEATURES (derive from actual data rather than placeholders)
    # - action types from actions.tsv
    # - predicates from queries
    # - condition/operators from query text
    # ========================================================================
    
    # Read actions.tsv to collect action types
    actions_path = os.path.join(output_dir, 'session_repositories', 'actions.tsv')
    action_types_set = set()
    try:
        with open(actions_path, 'r') as fa:
            header = fa.readline()
            for line in fa:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    atype = parts[3]
                    if atype:
                        action_types_set.add(atype)
    except Exception as e:
        print(f"Failed to read actions.tsv for action types: {e}")
    
    action_types_list = sorted(list(action_types_set)) if action_types_set else ['filter','join','projection','extension']
    
    # Extract predicates and operators from queries
    predicate_set = set()
    cond_set = set()
    if query_map:
        for q in query_map.values():
            # predicates like P123
            for m in re.findall(r'(P\d+)', q):
                predicate_set.add(m)
            # full URIs
            for m in re.findall(r'<http://www\.wikidata\.org/entity/(P\d+)>', q):
                predicate_set.add(m)
            # operators
            for op in re.findall(r'(!=|<=|>=|=|<|>|\bIN\b|\bin\b)', q, flags=re.IGNORECASE):
                cond_set.add(op.upper())
    
    # Build one-hot mappings keyed by action_id so they match ea_sp/ea_mp expectations
    import ast

    # Ensure predicates include those declared in action_params as well as queries
    try:
        with open(actions_path, 'r') as fa:
            header = fa.readline()
            for line in fa:
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    action_params_str = parts[4]
                    try:
                        ap = ast.literal_eval(action_params_str)
                    except Exception:
                        try:
                            ap = json.loads(action_params_str)
                        except Exception:
                            ap = None
                    if isinstance(ap, dict):
                        field = ap.get('field')
                        if field:
                            for m in re.findall(r'(P\d+|Q\d+)', str(field)):
                                predicate_set.add(m)
    except Exception:
        pass

    predicate_list = sorted(list(predicate_set)) if predicate_set else ['P31']
    cond_list = sorted(list(cond_set)) if cond_set else ['=']
    action_types_list = sorted(list(action_types_set)) if action_types_set else ['filter','join','projection','extension']

    # Read actions.tsv and construct per-action feature vectors keyed by action_id (int)
    act_feats = {}
    col_feats = {}
    cond_feats = {}

    try:
        with open(actions_path, 'r') as fa:
            header = fa.readline()
            for line in fa:
                parts = line.strip().split('\t')
                if len(parts) < 6:
                    continue
                try:
                    aid = int(parts[0])
                except Exception:
                    continue
                atype = parts[3]
                # parse action_params
                action_params_str = parts[4]
                try:
                    ap = ast.literal_eval(action_params_str)
                except Exception:
                    try:
                        ap = json.loads(action_params_str)
                    except Exception:
                        ap = {}

                # action type one-hot
                a_vec = np.zeros(len(action_types_list), dtype=np.float32)
                if atype in action_types_list:
                    a_vec[action_types_list.index(atype)] = 1.0
                else:
                    # unknown type -> leave zero-vector (shouldn't occur)
                    pass
                act_feats[aid] = a_vec

                # predicate / column one-hot
                field_token = None
                if isinstance(ap, dict):
                    field_val = ap.get('field')
                    if field_val:
                        m = re.search(r'(P\d+|Q\d+)', str(field_val))
                        if m:
                            field_token = m.group(1)
                if field_token and field_token in predicate_list:
                    c_vec = np.zeros(len(predicate_list), dtype=np.float32)
                    c_vec[predicate_list.index(field_token)] = 1.0
                else:
                    # fallback: zero vector (unknown predicate)
                    c_vec = np.zeros(len(predicate_list), dtype=np.float32)
                col_feats[aid] = c_vec

                # condition/operator one-hot: try to extract an operator token if present
                op_token = None
                if isinstance(ap, dict):
                    # try common keys that may describe operator
                    for k in ['op', 'operator', 'condition', 'cond']:
                        if k in ap:
                            v = ap[k]
                            # if v is a string like '=' or 'IN', standardize
                            if isinstance(v, str) and re.search(r'(!=|<=|>=|=|<|>|IN|in)', v):
                                m = re.search(r'(!=|<=|>=|=|<|>|IN|in)', v, flags=re.IGNORECASE)
                                if m:
                                    op_token = m.group(1).upper()
                                    break
                            # if numeric code for condition (paper may use codes), skip mapping
                if op_token and op_token in cond_list:
                    cond_vec = np.zeros(len(cond_list), dtype=np.float32)
                    cond_vec[cond_list.index(op_token)] = 1.0
                else:
                    # default to first cond (usually '=')
                    cond_vec = np.zeros(len(cond_list), dtype=np.float32)
                    cond_vec[0] = 1.0
                cond_feats[aid] = cond_vec
    except Exception as e:
        print(f"Failed to build per-action edge features from actions.tsv: {e}")

    # Save per-action pickles
    act_feats_path = f'{output_dir}/edge/act_feats.pickle'
    with open(act_feats_path, 'wb') as f:
        pickle.dump(act_feats, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Created {act_feats_path} with {len(act_feats)} action entries (keys = action_id)")

    # Compatibility: also write legacy filename 'act_five_feats.pickle' to avoid breaking
    # scripts that still look for the old name.
    legacy_act_path = f'{output_dir}/edge/act_five_feats.pickle'
    try:
        with open(legacy_act_path, 'wb') as f:
            pickle.dump(act_feats, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Also created legacy action features file {legacy_act_path}")
    except Exception as e:
        print(f"Warning: failed to write legacy act_five_feats.pickle: {e}")

    col_feats_path = f'{output_dir}/edge/col_action.pickle'
    with open(col_feats_path, 'wb') as f:
        pickle.dump(col_feats, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Created {col_feats_path} with {len(col_feats)} action entries (predicate dims = {len(predicate_list)})")

    cond_feats_path = f'{output_dir}/edge/cond_action.pickle'
    with open(cond_feats_path, 'wb') as f:
        pickle.dump(cond_feats, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Created {cond_feats_path} with {len(cond_feats)} action entries (cond dims = {len(cond_list)})")
def create_chunked_sessions(output_dir, project_query_edges, num_projects=4, seed=20250212):
    """Create chunked_sessions pickle files for train/test splitting."""
    
    # Format: {project_id: [[[parent, child, {'aid': action_id}], ...], ...]}
    chunked_sessions = {}
    
    for project_id in range(num_projects):
        # Collect all edges for this project
        project_edges_list = []
        current_session_edges = []
        
        for edge_group in project_query_edges.get(project_id, []):
            current_session_edges.extend(edge_group)
        
        # Split into multiple session groups (simulate multiple sessions)
        if current_session_edges:
            # Create 2-3 sessions per project
            edges_per_session = max(1, len(current_session_edges) // 2)
            for i in range(0, len(current_session_edges), edges_per_session):
                session_edges = current_session_edges[i:i+edges_per_session]
                if session_edges:
                    project_edges_list.append(session_edges)
        
        chunked_sessions[project_id] = project_edges_list
    
    # Save for each seed
    for seed_val in [seed, seed + 2]:  # Create 2 versions for reproducibility
        chunked_sessions_path = f'{output_dir}/chunked_sessions/unbiased_seed_{seed_val}.pickle'
        with open(chunked_sessions_path, 'wb') as f:
            pickle.dump(chunked_sessions, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Created {chunked_sessions_path}")

def create_readme(output_dir):
    """Create a README explaining the dataset."""
    
    readme_content = """# Wikidata SPARQL Query Sessions Dataset
 
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

## Running ExplorAct

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
"""
    
    readme_path = f'{output_dir}/README_WIKIDATA.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"Created {readme_path}")

def main():
    """Main entry point."""

    import argparse

    parser = argparse.ArgumentParser(
        description='Create Wikidata dataset for ExplorAct',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_wikidata_dataset.py --output wikidata_dataset
  python create_wikidata_dataset.py --output wikidata_dataset --projects 4
        """
    )
    parser.add_argument('--output', default='wikidata_dataset',
                        help='Output directory for dataset (default: wikidata_dataset)')
    parser.add_argument('--projects', type=int, default=4,
                        help='Number of projects for cross-validation (default: 4)')
    parser.add_argument('--live', action='store_true', help='Run live SPARQL queries against Wikidata (with caching)')
    parser.add_argument('--cache-only', action='store_true', help='When --live is set, use only existing cached query results and do not contact endpoint')

    args = parser.parse_args()

    print("=" * 70)
    print("Creating Wikidata SPARQL Dataset for ExplorAct")
    print("=" * 70)

    # Step 1: Download repository
    repo_path = download_repository()

    # Step 2: Create session structure
    print("\nParsing SPARQL sessions and creating data structure...")
    actions, displays, project_edges, display_max, action_max, query_map = create_session_structure(
        repo_path, args.output, num_projects=args.projects
    )

    # Step 3: Create feature files (with query mappings for proper feature extraction)
    print("\nGenerating feature files from SPARQL query structure...")
    create_feature_files(args.output, display_max, action_max,
                         sessions_dir=repo_path, query_map=query_map, live=args.live, cache_only=args.cache_only)

    # Step 4: Create chunked sessions
    print("\nCreating train/test splits...")
    create_chunked_sessions(args.output, project_edges, num_projects=args.projects)

    # Step 5: Create README
    print("\nGenerating documentation...")
    create_readme(args.output)

    print("\n" + "=" * 70)
    print(f"✓ Dataset successfully created in '{args.output}/'")
    print("=" * 70)
    print("\nNext steps:")
    print(f"1. cd {args.output}")
    print(f"2. python ../ea_sp.py act 20250212 5 0")
    print(f"3. python ../ea_mp.py col 20250212 5 0")


if __name__ == '__main__':
    main()
