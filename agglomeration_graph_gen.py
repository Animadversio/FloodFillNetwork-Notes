
import os
import time

import numpy as np
# import matplotlib.pylab as plt
import sys
# from absl import app
# from absl import flags
# from tensorflow import gfile
from google.protobuf import text_format
from ffn.inference import inference_pb2
from ffn.inference.resegmentation_pb2 import PairResegmentationResult
# from ffn.utils import bounding_box_pb2
# from ffn.inference import inference
# from ffn.inference import inference_flags
# from ffn.inference import resegmentation
from ffn.inference import resegmentation_analysis
# from analysis_script.utils_format_convert import read_image_vol_from_h5
# from ffn.inference.storage import subvolume_path
# from neuroglancer_segment_visualize import neuroglancer_visualize, GraphUpdater_show
# import neuroglancer
# from importlib import reload
from os.path import join
import glob
from contextlib import closing
import multiprocessing as mp
import logging
import networkx
import argparse
import ast
import pickle

logging.getLogger().setLevel(logging.INFO) # set the information level to show INFO logs

#%%
config='''inference {
    image {
      hdf5: "/scratch/binxu.wang/ffn-Data/LGN_DATA/grayscale_maps_LR.h5:raw"
    }
    image_mean: 136
    image_stddev: 55
    checkpoint_interval: 1800
    seed_policy: "PolicyPeaks"
    model_checkpoint_path: "/scratch/binxu.wang/ffn-Data/models/LR_model_Longtime_Mov/model.ckpt-11932826"
    model_name: "convstack_3d.ConvStack3DFFNModel"
    model_args: "{\\"depth\\": 9, \\"fov_size\\": [37, 25, 15], \\"deltas\\": [8,6,2]}"
    segmentation_output_dir: "/scratch/binxu.wang/results/LGN/testing_exp12/"
    inference_options {
      init_activation: 0.95
      pad_value: 0.05
      move_threshold: 0.9
      min_boundary_dist { x: 5 y: 5 z: 1}
      segment_threshold: 0.6
      min_segment_size: 5000
      disco_seed_threshold: 0.002
    }
    init_segmentation {
       npz: "/scratch/binxu.wang/results/LGN/testing_exp12/0/0/seg-0_0_0.npz:segmentation"
    }
}
radius {x: 50 y: 50 z: 17}
output_directory: "/scratch/binxu.wang/results/LGN/testing_exp12/reseg"
max_retry_iters: 10
segment_recovery_fraction: 0.6
analysis_radius {x: 35 y: 35 z: 10}
'''
#reseg_dir = "/home/morganlab/Downloads/ffn-master/results/LGN/testing_exp12/reseg" # reseg_req.output_directory
# seg_dir = subvolume_path(reseg_req.inference.segmentation_output_dir, (0, 0, 0), 'npz')
# seg_dir = subvolume_path("/home/morganlab/Downloads/ffn-master/results/LGN/testing_exp12", (0, 0, 0), 'npz')

#%%
ap = argparse.ArgumentParser()
ap.add_argument(
    '--config',
    help='config proto of resegment')
ap.add_argument(
    '--config_path',
    help='path of config proto file of resegment')
ap.add_argument(
    '--seg_dir',
    help='path of point segmentation')
ap.add_argument(
    '--reseg_dir',
    help='path of point list proto file of resegment')
ap.add_argument(
    '--output_dir',
    help='path of point list proto file of resegment')
ap.add_argument(
    '--pixelsize', help='pixelsize tuple in x,y,z order in nm')
#%%
args = ap.parse_args()
if args.config:
    config = args.config
elif args.config_path:
    f = open(args.config_path, "r")
    config = f.read()
    f.close()
else:
    config = config
reseg_req = inference_pb2.ResegmentationRequest()
_ = text_format.Parse(config, reseg_req)
req = reseg_req.inference
if args.seg_dir:
    seg_dir = args.seg_dir
else:
    seg_dir = req.init_segmentation.npz.split(':')[0]
if args.reseg_dir:
    reseg_dir = args.reseg_dir
else:
    reseg_dir = reseg_req.output_directory
if args.output_dir:
    output_dir = args.output_dir
else:
    output_dir = reseg_dir
os.makedirs(output_dir, exist_ok=True)
# [30, 12, 8]
if args.pixelsize:
    pixelsize = ast.literal_eval(args.pixelsize)
    voxelsize_zyx = list(pixelsize[::-1])
else:
    voxelsize_zyx = [40, 8, 8]
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% Evaluation of resegmentation network
reseg_r_zyx = [reseg_req.radius.z, reseg_req.radius.y, reseg_req.radius.x]
analysis_r_zyx = [reseg_req.analysis_radius.z, reseg_req.analysis_radius.y, reseg_req.analysis_radius.x]
f = np.load(seg_dir)
seg = f['segmentation'] # load the segmentation for evaluation
f.close()
#%%
#%% Multiprocessor version
from ffn.inference.resegmentation_analysis import IncompleteResegmentationError, InvalidBaseSegmentatonError
from zipfile import BadZipFile
# t0 = time.time()
# for filename in savefile_list:
#     try:
#         result_proto = resegmentation_analysis.evaluate_pair_resegmentation(join(reseg_dir, filename),
#                                 seg, reseg_r_zyx, analysis_r_zyx, sampling=voxelsize_zyx)
#         proto_list.append(result_proto)
#         proto_string_list.append(result_proto.SerializeToString())
#         segment_graph.add_weighted_edges_from([(result_proto.id_a, result_proto.id_b, result_proto.eval.iou)])
#     except IncompleteResegmentationError:
#         logging.info("Resegmentation incomplete error in file %s." % filename)
#     except InvalidBaseSegmentatonError:
#         logging.info("Invalid Base Segmentation error in file %s." % filename)
#     except BadZipFile:
#         logging.warning("This zip file %s is broken, raise magic number error ." % filename)
#     except:
#         logging.warning("Some other error happened!! %s" % sys.exc_info()[0])
#
# print("Spent all", time.time()-t0, 's')
#%%
def worker_func(filename):
    global seg, reseg_r_zyx, analysis_r_zyx, voxelsize_zyx
    try:
        result_proto = resegmentation_analysis.evaluate_pair_resegmentation(filename,
                                seg, reseg_r_zyx, analysis_r_zyx, sampling=voxelsize_zyx)
        return result_proto.SerializeToString()
    except IncompleteResegmentationError:
        logging.info("Resegmentation incomplete error in file %s." % filename.split('/')[-1])
    except InvalidBaseSegmentatonError:
        logging.info("Invalid Base Segmentation error in file %s." % filename.split('/')[-1])
    except BadZipFile:
        logging.warning("This zip file %s is broken, raise magic number error ." % filename.split('/')[-1])
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except:
        logging.warning("Some other error happened!! %s" % sys.exc_info()[0])
    return
#%%
savefile_list = glob.glob(join(reseg_dir,"*.npz"))
raw_cnt = len(savefile_list)
print("Total %d files to be processed" % raw_cnt)
# savefile_list = os.listdir(reseg_dir)
worker_n = mp.cpu_count()  # the code above does not work in Python 2.x but do in 3.6
with closing(mp.Pool(processes=worker_n)) as pool: #  mp.cpu_count()//2
    result = pool.imap_unordered(worker_func, savefile_list, chunksize=100)  # returns a generator _unordered
print("Started making the pool")
#%%
segment_graph = networkx.Graph()
proto_list = []
proto_string_list = []
idx = np.unique(seg)
idx = idx[1:]  # leave out background
segment_graph.add_nodes_from(idx)
proto_cnt = 0
raw_cnt = len(savefile_list)
for result_proto_str in result:
    if not result_proto_str is None:
        proto_cnt += 1
        # print(proto_cnt, '\n')
        proto_string_list.append(result_proto_str)
        result_proto = PairResegmentationResult.FromString(result_proto_str)    
        # proto_list.append(result_proto)
        segment_graph.add_weighted_edges_from([(result_proto.id_a, result_proto.id_b, result_proto.eval.iou)])
        if proto_cnt % 100 == 0:
            print("Processed %d/%d (%.1f /100)" % (proto_cnt, raw_cnt, 100.0*proto_cnt/raw_cnt))
#%%
pickle.dump(proto_string_list, open(join(output_dir, "proto_summary.pkl"), "wb"))
pickle.dump(segment_graph, open(join(output_dir, "segment_graph.pkl"), "wb"))
#%%
strong_edge = [(u, v) for (u, v, d) in segment_graph.edges(data=True) if d['weight'] > 0.8]  # filter the edges here!!!!
connect_segment_graph = networkx.Graph()
connect_segment_graph.add_nodes_from(segment_graph.nodes)
connect_segment_graph.add_edges_from(strong_edge)
pickle.dump(connect_segment_graph, open(join(output_dir, "connect_segment_graph.pkl"), "wb"))
#%%
pool.close()
print("closed pool")
print("joining pool")
pool.join()
print("joined pool")
#%%
reseg_dir =  "/home/morganlab/Downloads/ffn-master/results/LGN/testing_exp12/reseg"
import pickle
segment_graph = pickle.load(open(join(reseg_dir, "segment_graph.pkl"), "rb"))
# strong_edge = [(u, v) for (u, v, d) in segment_graph.edges(data=True) if d['weight'] > 0.8]  # filter the edges here!!!!
# connect_segment_graph = networkx.Graph()
# connect_segment_graph.add_nodes_from(segment_graph.nodes)
# connect_segment_graph.add_edges_from(strong_edge)
# #%%
# #%%
# for component in networkx.connected_components(connect_segment_graph):
#     if len(component) > 1:
#         print(component)
#
# #%%
# # result_proto = resegmentation_analysis.evaluate_pair_resegmentation("/home/morganlab/Documents/Autoseg_result/Autoseg_exp7/reseg/120-1279_at_672_582_92.npz",
# #                                                                     seg, [20, 200, 200], [20, 200, 200], sampling=[30, 12, 8])
# # #%%
# # result_proto2 = resegmentation_analysis.evaluate_pair_resegmentation("/home/morganlab/Documents/Autoseg_result/Autoseg_exp7/reseg/1279-1235_at_689_564_92.npz",
# #                                                                     seg, [20, 200, 200], [20, 200, 200], sampling=[30, 12, 8])
#
#
# #%% In situ visualization
# seg_dict = {
#             "seg_12": {"seg_dir": "/home/morganlab/Downloads/ffn-master/results/LGN/testing_exp12"},
#             }
# image_dir = "/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/grayscale_maps_LR.h5"
# # viewer = neuroglancer_visualize(seg_dict, image_dir)
# #%%
# graph_undater = GraphUpdater_show(segment_graph, list(segment_graph.nodes), [], seg_dict, image_dir)
