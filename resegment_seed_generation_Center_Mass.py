import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import multiprocessing as mp
from contextlib import closing
import ctypes
import os
from os.path import join
import argparse
from analysis_script.utils_format_convert import subvolume_path
import resource
import scipy.ndimage as ndimage
import ast
from ffn.utils import bounding_box_pb2
from google.protobuf import text_format
#%%
memory_check = False
metric = [8, 8, 40]  # voxel size in x,y,z order in nm
dist_threshold = 50  # maximum distance in nm to be considered valid pair
threshold = 50  # minimum overlap to be consider a pair
move_vec = (5, 5, 3)
size = (152, 550, 550)
corner = (0, 0, 0)
offset = (0, 0, 0)
worker_n = mp.cpu_count()
seg_path = "/home/morganlab/Documents/ixP11LGN/p11_6_consensus_33_38_full"
# "/Users/binxu/Connectomics_Code/results/LGN/testing_exp12"
# "/home/morganlab/Downloads/ffn-master/results/LGN/testing_exp12/"
# "/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR/"
# "/home/morganlab/Downloads/ffn-master/results/LGN/testing_exp12/"
# "/Users/binxu/Connectomics_Code/results/LGN/testing_LR/0/0/"
# "/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR/0/0/"
# "/Users/binxu/Connectomics_Code/results/LGN/"
# '/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR/0/0/'
output_path = "/home/morganlab/Documents/ixP11LGN/p11_6_consensus_33_38_full/"
# '/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR/'
# "/home/morganlab/Downloads/ffn-master/results/LGN/testing_exp12/"
# "/scratch/binxu.wang/ffn-Data/results/LGN/testing_LR/"
# "/Users/binxu/Connectomics_Code/results/LGN/testing_LR/"

ap = argparse.ArgumentParser()
ap.add_argument(
    '--seg_path', help='')
ap.add_argument(
    '--output_path', help='Output the files')
ap.add_argument(
    '--worker_n', help='Output the files')
ap.add_argument(
    '--corner', help='the corner used in the path to fetch segmentation')
ap.add_argument(
    '--size', help='')
ap.add_argument(
    '--offset', help='the starting corner in subvolume of seg (currently w.r.t the corner, relative cordinate  )')
ap.add_argument(
    '--bounding_box', help='Bounding box setting will overwrite the offset and size setting ')

args = ap.parse_args()
if args.seg_path:
    seg_path = args.seg_path
if args.corner:
    corner = ast.literal_eval(args.corner)
else:
    corner = (0, 0, 0)
if args.output_path:
    output_path = args.output_path
elif args.seg_path:
    output_path = args.seg_path
if args.worker_n:
    worker_n = args.worker_n
if args.bounding_box:
    bbox = bounding_box_pb2.BoundingBox()  # bounding box structure
    text_format.Parse(args.bounding_box, bbox)
    offset = (bbox.start.z, bbox.start.y, bbox.start.x)
    size = (bbox.size.z, bbox.size.y, bbox.size.x)
else:
    if args.offset:
        offset = ast.literal_eval(args.offset)
    else:
        offset = (0, 0, 0)
    if args.size:
        size = ast.literal_eval(args.size)
    else:
        size = None
#%%
metric = np.array(metric)
metric = metric.reshape((-1, 1))  # reshape to ensure the computation below
metric = metric[::-1]  # in z y x order
for path in [output_path]:
    os.makedirs(path, exist_ok=True)
#%%
print('[%d] At start Memory usage: %s (kb)' % (os.getpid(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
data = np.load(subvolume_path(seg_path, corner, 'npz'))
segmentation = data['segmentation']
data.close()
if not size==None:
    segmentation = segmentation[offset[0]:offset[0]+size[0], offset[1]:offset[1]+size[1], offset[2]:offset[2]+size[2]]
    if np.any([offset[i]+size[i] > segmentation.shape[i] for i in range(3)]):
        print("Warning: fetching subvolume out of bound!!!")
segmentation = segmentation.astype(np.int)  # make sure full byte width, or BASE * will outflow
print('[%d] After cast type Memory usage: %s (kb)' %
      (os.getpid(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
#%%
vx, vy, vz = move_vec
BASE = segmentation.max() + 1
def _sel(i):
    if i == 0:
        return slice(None)
    elif i > 0:
        return slice(i, None)
    else:
        return slice(None, i)
composite_map = segmentation[_sel(-vz), _sel(-vx), _sel(-vy)] + BASE * segmentation[_sel(vz), _sel(vx), _sel(vy)]

pair_idx, pair_cnts=np.unique(composite_map, return_counts=True)
idx2, idx1 = np.divmod(pair_idx, BASE)
pair_array = np.array([idx1, idx2]).T

def symmetrize_pair_array(pair_array, pair_cnts):
    pair_array_sym = np.sort(pair_array, axis=1)
    pair_array_sym = np.unique(pair_array_sym,axis=0)
    pair_idx_sym = pair_array_sym[:, 0] + pair_array_sym[:, 1]*BASE
    pair_cnts_sym = np.zeros(pair_array_sym.shape[0])
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
    return pair_array_sym, pair_cnts_sym

pair_array_sym, pair_cnts_sym = symmetrize_pair_array(pair_array, pair_cnts)
assert pair_cnts_sym.sum() == pair_cnts.sum()
# Threshold of overlap size can be added !
#%%
def worker_func(id_pair):
    global composite_map, BASE, metric, segmentation
    # id_pair = pair_array_sym[1]
    cur_idx1, cur_idx2 = id_pair[0], id_pair[1]
    if cur_idx1 == cur_idx2 or cur_idx1 * cur_idx2 == 0:
        return []  # ignore the overlap with background and samething overlap
    # seg_a = segmentation == cur_idx1
    # seg_b = segmentation == cur_idx2
    if memory_check:
        print('[%d] After calculate segment Memory usage: %s (kb)' % (os.getpid(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    composite_mask = composite_map == cur_idx1 + BASE * cur_idx2
    if not composite_mask.sum() == 0:
        index_list = np.array(composite_mask.nonzero())
        com1 = index_list.mean(axis=1) + np.array([vz, vy, vx])//2
        # full_mask = np.zeros(segmentation.shape, dtype=np.bool)
        # full_mask[_sel(-vz), _sel(-vx), _sel(-vy)] += composite_mask
        # full_mask[_sel(vz), _sel(vx), _sel(vy)] += composite_mask
        # com1 = ndimage.measurements.center_of_mass(full_mask)
        com1 = [int(i) for i in com1]
    else:
        com1 = None
    composite_mask  = composite_map == cur_idx2 + BASE * cur_idx1
    if not composite_mask.sum() == 0:
        # full_mask = np.zeros(segmentation.shape, dtype=np.bool)
        # full_mask[_sel(-vz), _sel(-vx), _sel(-vy)] += composite_mask
        # full_mask[_sel(vz), _sel(vx), _sel(vy)] += composite_mask
        # com2 = ndimage.measurements.center_of_mass(full_mask)
        index_list = np.array(composite_mask.nonzero())
        com2 = index_list.mean(axis=1) + np.array([vz, vy, vx]) // 2
        com2 = [int(i) for i in com2]
    else:
        com2 = None
    print("{id_a:%d id_b:%d point {%s} } \n" % (cur_idx1, cur_idx2, str([com1, com2])))
    if memory_check:
        print('[%d] After calculate center Memory usage: %s (kb)' % (os.getpid(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    return [com1, com2]

#%%
valid_mask = (pair_array_sym[:,0]!=pair_array_sym[:,1]) * \
             (pair_array_sym[:,0]*pair_array_sym[:,1]!=0) * \
             (pair_cnts_sym > threshold)
pair_num = sum(valid_mask)
print("Pairs to process %d." % (pair_num))  # exclude background and same type
pair_array_sym = pair_array_sym[valid_mask, :]
pair_cnts_sym = pair_cnts_sym[valid_mask]

#%% parallelize the program
if memory_check:
    print('[%d] Before starting Pool Memory usage: %s (kb)' %
      (os.getpid(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
pair_list = list(pair_array_sym)
# mp.cpu_count())  # the code above does not work in Python 2.x but do in 3.6
with closing(mp.Pool(processes=worker_n)) as pool: #  mp.cpu_count()//2
    result = pool.imap(worker_func, pair_list, chunksize=50)  # returns a generator _unordered
# result_list = list(result)
# pickle.dump(result_list, open(join(output_path, 'seed_result.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)

if memory_check:
    print('[%d] Before writing down Memory usage: %s (kb)' %
      (os.getpid(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
# Save result to dict
seed_dict = {}
for result_vec, id_pair in zip(result, pair_list):
    print(id_pair)
    cur_idx1, cur_idx2 = id_pair[0], id_pair[1]
    for vec in result_vec:
        if not vec==None:
            vec = [vec[i] + corner[i] + offset[i] for i in range(3)]  # translate into global coordinate for the volume
            seed_dict[(cur_idx1, cur_idx2)] = seed_dict.get((cur_idx1, cur_idx2), []) + [vec]
# Write the pb file
reseg_corner = [corner[i] + offset[i] for i in range(3)]
pickle.dump(seed_dict, open(join(output_path, 'seed_dict_%d_%d_%d.pkl' % (reseg_corner[2], reseg_corner[1], reseg_corner[0])), 'wb'), pickle.HIGHEST_PROTOCOL)  # should store more information into pkl file
output_fn = "point_list_%d_%d_%d.txt" % (reseg_corner[2], reseg_corner[1], reseg_corner[0])
file = open(join(output_path, output_fn), "w")
for pair in seed_dict:
    for pnt in seed_dict[pair]:
        file.write("points {id_a:%d id_b:%d point {x: %d y: %d z: %d} } \n" % (pair[0], pair[1], pnt[2], pnt[1], pnt[0]))
file.close()

pool.close()
print("closed pool")
print("joining pool")
pool.join()
print("joined pool")

