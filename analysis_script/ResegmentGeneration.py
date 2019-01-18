#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 14:52:18 2019

@author: morganlab

Search for connecting components in a object tensor 
"""


vx, vy, vz = (10, 10, 0)
BASE = segmentation.max() + 1
def _sel(i):
    if i == 0:
        return slice(None)
    elif i > 0:
        return slice(i, None)
    else:
        return slice(None, i)
composite_map = segmentation[_sel(-vz), _sel(-vx), _sel(-vy)].astype(np.int) + BASE * segmentation[_sel(vz), _sel(vx), _sel(vy)].astype(np.int)
pair_idx, pair_cnts=np.unique(composite_map,return_counts=True)
idx2, idx1 = np.divmod(pair_idx, BASE)
pair_array = np.array([idx1, idx2]).T
pair_array_sym =np.sort(pair_array, axis=1)
pair_array_sym = np.unique(pair_array_sym,axis=0)
pair_idx_sym = pair_array_sym[:, 0] + pair_array_sym[:, 1]*BASE
pair_cnts_sym =np.zeros(pair_array_sym.shape[0])
for i in range(len(pair_cnts)):
    relid1 = np.where(pair_idx_sym==(idx1[i] +BASE*idx2[i]))[0]
    relid2 = np.where(pair_idx_sym==(idx2[i] +BASE*idx1[i]))[0]
    if len(relid1)==0:
        pair_cnts_sym[relid2] += pair_cnts[i]
    elif len(relid2)==0:
        pair_cnts_sym[relid1] += pair_cnts[i]
    else:  # same index idx1[i]==idx2[i]
        assert relid1==relid2
        pair_cnts_sym[relid2] += pair_cnts[i]

# Threshold of overlap size can be added ! 