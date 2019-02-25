
import os
import time

import numpy as np
import matplotlib.pylab as plt
import sys
from absl import app
from absl import flags
from tensorflow import gfile
from google.protobuf import text_format
from ffn.inference import inference
from ffn.inference import inference_pb2

from ffn.utils import bounding_box_pb2
from ffn.inference import inference
from ffn.inference import inference_flags
from ffn.inference import resegmentation
from ffn.inference import resegmentation_analysis

from analysis_script.utils_format_convert import read_image_vol_from_h5
from ffn.inference.storage import subvolume_path
from neuroglancer_segment_visualize import neuroglancer_visualize
import neuroglancer
from importlib import reload
from os.path import join
import logging
import networkx
logging.getLogger().setLevel(logging.INFO) # set the information level to show INFO logs


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
reseg_req = inference_pb2.ResegmentationRequest()
_ = text_format.Parse(config, reseg_req)
req = reseg_req.inference

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% Evaluation of resegmentation network



reseg_r_zyx = [reseg_req.radius.z, reseg_req.radius.y, reseg_req.radius.x]
analysis_r_zyx = [reseg_req.analysis_radius.z, reseg_req.analysis_radius.y, reseg_req.analysis_radius.x]
# seg_dir = subvolume_path(reseg_req.inference.segmentation_output_dir, (0, 0, 0), 'npz')
seg_dir = subvolume_path("/home/morganlab/Downloads/ffn-master/results/LGN/testing_exp12", (0, 0, 0), 'npz')
voxelsize_zyx = [30, 12, 8]

f = np.load(seg_dir)
seg = f['segmentation'] # load the segmentation for evaluation
f.close()
#%%
segment_graph = networkx.Graph()
proto_list = []
reseg_dir = "/home/morganlab/Downloads/ffn-master/results/LGN/testing_exp12/reseg" # reseg_req.output_directory
savefile_list = os.listdir(reseg_dir)
idx = np.unique(seg)
idx = idx[1:]
segment_graph.add_nodes_from(idx)
#%%
from ffn.inference.resegmentation_analysis import IncompleteResegmentationError, InvalidBaseSegmentatonError
from zipfile import BadZipFile
t0 = time.time()
for filename in savefile_list:
    try:
        result_proto = resegmentation_analysis.evaluate_pair_resegmentation(join(reseg_dir, filename),
                                seg, reseg_r_zyx, analysis_r_zyx, sampling=voxelsize_zyx)
        proto_list.append(result_proto)
        segment_graph.add_weighted_edges_from([(result_proto.id_a, result_proto.id_b, result_proto.eval.iou)])
    except IncompleteResegmentationError:
        logging.info("Resegmentation incomplete error in file %s." % filename)
    except InvalidBaseSegmentatonError:
        logging.info("Invalid Base Segmentation error in file %s." % filename)
    except BadZipFile:
        logging.warning("This zip file %s is broken, raise magic number error ." % filename)
    except:
        logging.warning("Some other error happened!! %s" % sys.exc_info()[0])

print(time.time()-t0, 's')
#%%
#%%
strong_edge = [(u, v) for (u, v, d) in segment_graph.edges(data=True) if d['weight'] > 0.5]  # filter the edges here!!!!
connect_segment_graph = networkx.Graph()
connect_segment_graph.add_nodes_from(idx)
connect_segment_graph.add_edges_from(strong_edge)
#%%
import pickle
# pickle.dump(result_proto, open(join(reseg_dir, "proto_summary.pkl"), "wb"))

pickle.dump(segment_graph, open(join(reseg_dir, "segment_graph.pkl"), "wb"))

#%%
for component in networkx.connected_components(connect_segment_graph):
    if len(component) > 1:
        print(component)




#%%
# result_proto = resegmentation_analysis.evaluate_pair_resegmentation("/home/morganlab/Documents/Autoseg_result/Autoseg_exp7/reseg/120-1279_at_672_582_92.npz",
#                                                                     seg, [20, 200, 200], [20, 200, 200], sampling=[30, 12, 8])
# #%%
# result_proto2 = resegmentation_analysis.evaluate_pair_resegmentation("/home/morganlab/Documents/Autoseg_result/Autoseg_exp7/reseg/1279-1235_at_689_564_92.npz",
#                                                                     seg, [20, 200, 200], [20, 200, 200], sampling=[30, 12, 8])


#%% In situ visualization
seg_dict = {
            "seg_12": {"seg_dir": "/home/morganlab/Downloads/ffn-master/results/LGN/testing_exp12"},
            }
image_dir = "/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/grayscale_maps_LR.h5"
viewer = neuroglancer_visualize(seg_dict, image_dir)
