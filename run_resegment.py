"""Runs FFN inference within a dense bounding box.

Inference is performed within a single process.
"""

import os
import time

import numpy as np
import matplotlib.pylab as plt

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
from importlib import reload
import logging
import argparse
import ast
logging.getLogger().setLevel(logging.INFO) # set the information level to show INFO logs
# # reload(inference)
# reload(resegmentation_analysis)
#%% default example of resegmentation protobuf
config = """inference {
        image {
          hdf5: "/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/grayscale_maps_LR(copy).h5:raw"
        }
        image_mean: 128
        image_stddev: 33
        checkpoint_interval: 1800
        seed_policy: "PolicyPeaks"
        model_checkpoint_path: "/home/morganlab/Downloads/ffn-master/models/LR_model_Longtime/model.ckpt-4697585"
        model_name: "convstack_3d.ConvStack3DFFNModel"
        model_args: "{\\"depth\\": 9, \\"fov_size\\": [55, 37, 17], \\"deltas\\": [9,6,3]}"
        segmentation_output_dir: "/home/morganlab/Downloads/ffn-master/results/LGN/testing_exp7/"
        inference_options {
          init_activation: 0.95
          pad_value: 0.05
          move_threshold: 0.9
          min_boundary_dist { x: 5 y: 5 z: 1}
          segment_threshold: 0.6
          min_segment_size: 5000
          disco_seed_threshold: 0.005
        }
        init_segmentation {
           npz: "/home/morganlab/Downloads/ffn-master/results/LGN/testing_exp7/0/0/seg-0_0_0.npz:segmentation"
        }
    }
    points {id_a:120 id_b:1279 point {x: 672 y: 582 z: 92} }
    points {id_a:1279 id_b:1235 point {x: 689 y: 564 z: 92} }
    radius {x: 200 y: 200 z: 20}
    output_directory: "/home/morganlab/Documents/Autoseg_result/Autoseg_exp7/reseg"
    max_retry_iters: 2
    segment_recovery_fraction: 0.4
    analysis_radius {x: 200 y: 200 z: 20}
    """

#%%
ap = argparse.ArgumentParser()
ap.add_argument(
    '--config',
    help='config proto of resegment')
ap.add_argument(
    '--config_path',
    help='path of config proto file of resegment')
ap.add_argument(
    '--point_path',
    help='path of point list proto file of resegment')
ap.add_argument(
    '--pixelsize', help='pixelsize tuple in x,y,z order in nm')
#%%
if __name__=="__main__":
    pixelsize = (8, 12, 30)
    args = ap.parse_args()
    if args.config:
        config = args.config
    elif args.config_path:
        f = open(args.config_path, "r")
        config = f.read()
        f.close()
    if args.point_path:
        f = open(args.point_path, "r")
        point_list = f.read()
        f.close()
        config += '\n' + point_list
    if args.pixelsize:
        pixelsize = ast.literal_eval(args.pixelsize)
    reseg_req = inference_pb2.ResegmentationRequest()
    _ = text_format.Parse(config, reseg_req)
    req = reseg_req.inference
    # req = inference_pb2.InferenceRequest()
    # _ = text_format.Parse(config, req)
    #%%
    runner = inference.Runner()
    runner.start(req)
    runner.set_pixelsize(pixelsize)
    # seg_canvas1 = runner.run((0, 0, 0), (1000, 1000, 175))
    resegmentation.process(reseg_req, runner)