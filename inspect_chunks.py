#!/usr/bin/env python3
import pickle

with open('./wikidata_dataset/chunked_sessions/unbiased_seed_20250212.pickle', 'rb') as f:
    data = pickle.load(f)

print('Keys (project IDs):', list(data.keys()))
for k in data:
    session_count = len(data[k])
    edge_count = sum(len(edges) for edges in data[k])
    print(f'  Project {k}: {session_count} session groups, total edges: {edge_count}')
