# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs FFN inference
This script aims at run inference from several preselected seeds
and inspect the single seed segemtation result.
"""

import os
import time

from google.protobuf import text_format
from absl import app
from absl import flags
from tensorflow import gfile
import numpy as np
import sys
import logging
import logging.config
# logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)
# logging.config.fileConfig(json.load(open('configs/logging.json')), disable_existing_loggers=False)
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(filename)s: %(funcName)s(): %(lineno)d] %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level":"INFO",
            "class":"logging.StreamHandler",
        },
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "logfile": {
            "class": "logging.FileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": "inference_log_new.log",
            "encoding": "utf8"
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "logfile"],
            "level": "INFO",
            "propagate": True
        }
    }
})
logging.info("Logger prepared! ")
# handlers = [logging.StreamHandler(sys.stdout),
#             logging.FileHandler('/home/morganlab/PycharmProjects/FloodFillNetwork-Notes/inference_log3.log'), ]
# logging.basicConfig(level=logging.INFO, handlers=handlers, disable_existing_loggers=False,
#                     format=('%(asctime)s [%(filename)s: %(funcName)s(): %(lineno)d] %(message)s'))

from ffn.utils import bounding_box_pb2
from ffn.inference import inference
from ffn.inference import inference_pb2
from ffn.inference import storage
from scipy.special import expit, logit

corner = (0,0,0)
downsample_factor =1
#%%
# Model on LGN of mice
# "models/LR_model_Longtime/model.ckpt-264380"
config = '''image {
 hdf5: "/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/grayscale_maps_LR.h5:raw"
}
image_mean: 136
image_stddev: 55
checkpoint_interval: 1200
seed_policy: "PolicyPeaks"
model_checkpoint_path: "/home/morganlab/Downloads/ffn-master/models/LR_model_Longtime/model.ckpt-1127307"
model_name: "convstack_3d.ConvStack3DFFNModel"
model_args: "{\\"depth\\": 9, \\"fov_size\\": [55, 37, 17], \\"deltas\\": [9,6,3]}"
segmentation_output_dir: "/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR_Longtime_point3"
inference_options {
  init_activation: 0.95
  pad_value: 0.05
  move_threshold: 0.90
  min_boundary_dist { x: 5 y: 5 z: 1}
  segment_threshold: 0.6
  min_segment_size: 1000
  disco_seed_threshold: 0.005
}'''
seed_list = [(1080, 860, 72), (1616, 1872, 43), (612, 1528, 92), (616, 180, 92),  (144, 712, 43), (400, 168, 45), (1332, 248, 45), (120, 700,45)]  # in xyz order
downsample_factor =2
canvas_bbox = [(0, 0, 0), (175, 1058, 1180)]

#%%
request = inference_pb2.InferenceRequest()
_ = text_format.Parse(config, request)
if not gfile.Exists(request.segmentation_output_dir):
    gfile.MakeDirs(request.segmentation_output_dir)
runner = inference.Runner()
runner.start(request)
canvas, alignment = runner.make_canvas(canvas_bbox[0], canvas_bbox[1]) # like (0, 0, 0), (175, 1058, 1180)
for id, start_point in enumerate(seed_list):
    label = id + 1
    pos = (int(start_point[2]-corner[2]), int((start_point[1]-corner[1])//downsample_factor), int((start_point[0]-corner[0])//downsample_factor))
    canvas.log_info('Starting segmentation at %r (zyx)', pos)
    num_iters = canvas.segment_at(pos,)  # zyx
                    # dynamic_image=inference.DynamicImage(),
                    # vis_update_every=2, vis_fixed_z=True)
    mask = canvas.seed >= request.inference_options.segment_threshold

    raw_segmented_voxels = np.sum(mask)  # count the number of raw segmented voxels
    # Record existing segment IDs overlapped by the newly added object.
    overlapped_ids, counts = np.unique(canvas.segmentation[mask],
                                       return_counts=True)
    valid = overlapped_ids > 0  # id < 0 are the background and invalid voxels
    overlapped_ids = overlapped_ids[valid]
    counts = counts[valid]
    if len(counts) > 0:
        print('Overlapping segments (label, size): ', *zip(overlapped_ids, counts), sep=" ")

    mask &= canvas.segmentation <= 0  # get rid of the marked voxels from mask
    actual_segmented_voxels = np.sum(mask)  # only count those unmarked voxel in `actual_segmented_voxels`
    canvas.segmentation[mask] = label
    canvas.log_info('Created supervoxel:%d  seed(zyx):%s  size:%d (raw size %d)  iters:%d',
                label, pos, actual_segmented_voxels, raw_segmented_voxels, num_iters)

    np.savez(os.path.join(request.segmentation_output_dir, 'seg%03d_%s.npz' % (label, str(start_point))),
           segmentation=mask,
            prob=storage.quantize_probability(
            expit(canvas.seed)))

counter_path = os.path.join(request.segmentation_output_dir, 'counters.txt')
if not gfile.Exists(counter_path):
    runner.counters.dump(counter_path)
 #%%
corner = (0,0,0)
seg_path = storage.segmentation_path(
        request.segmentation_output_dir, corner)
prob_path = storage.object_prob_path(
        request.segmentation_output_dir, corner)
runner.save_segmentation(canvas, alignment, seg_path, prob_path)







