#!/usr/bin/env python3
import pickle
import numpy as np

# Check display features
with open('./wikidata_dataset/display_feats/display_pca_feats_9999.pickle', 'rb') as f:
    display_feats = pickle.load(f)
sample_display = next(iter(display_feats.values()))
print(f'Display features: {len(display_feats)} displays, dim={sample_display.shape}')

# Check edge features
with open('./wikidata_dataset/edge/act_feats.pickle', 'rb') as f:
    act_feats = pickle.load(f)
sample_act = next(iter(act_feats.values()))
print(f'Action features: {len(act_feats)} actions, dim={sample_act.shape}')

with open('./wikidata_dataset/edge/col_action.pickle', 'rb') as f:
    col_feats = pickle.load(f)
sample_col = next(iter(col_feats.values()))
print(f'Column features: {len(col_feats)} actions, dim={sample_col.shape}')

with open('./wikidata_dataset/edge/cond_action.pickle', 'rb') as f:
    cond_feats = pickle.load(f)
sample_cond = next(iter(cond_feats.values()))
print(f'Condition features: {len(cond_feats)} actions, dim={sample_cond.shape}')

# concat_feats would be act + col
concat_dim = sample_act.shape[0] + sample_col.shape[0]
print(f'\nconcat_feats dimension (act + col): {concat_dim}')
print(f'Expected edge_dim in ea_sp.py: EDGE_DIM dynamically computed = {concat_dim}')
