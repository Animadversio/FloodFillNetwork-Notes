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
Inference is performed within a single process.
"""

import os
import time

from google.protobuf import text_format
from absl import app
from absl import flags
from tensorflow import gfile

from ffn.utils import bounding_box_pb2
from ffn.inference import inference
from ffn.inference import inference_flags
import logging
logging.getLogger().setLevel(logging.INFO)
#%%
config = '''image {
 hdf5: "/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/grayscale_maps_LR.h5:raw"
}
image_mean: 136
image_stddev: 55
checkpoint_interval: 1200
seed_policy: "PolicyPeaks"
model_checkpoint_path: "models/LR_model_Longtime/model.ckpt-230168"
model_name: "convstack_3d.ConvStack3DFFNModel"
model_args: "{\"depth\": 9, \"fov_size\": [55, 37, 17], \"deltas\": [9,6,3]}"
segmentation_output_dir: "results/LGN/testing_LR_Longtime2"
inference_options {
  init_activation: 0.95
  pad_value: 0.05
  move_threshold: 0.85
  min_boundary_dist { x: 5 y: 5 z: 1}
  segment_threshold: 0.6
  min_segment_size: 1000
}'''

request = inference_pb2.InferenceRequest()
_ = text_format.Parse(config, request)
def main(unused_argv):

  if not gfile.Exists(request.segmentation_output_dir):
    gfile.MakeDirs(request.segmentation_output_dir)

  bbox = bounding_box_pb2.BoundingBox()  # bounding box structure
  text_format.Parse(FLAGS.bounding_box, bbox)  # Parse the param in flag

  runner = inference.Runner()
  runner.start(request)
  canvas, alignment = runner.make_canvas((0, 0, 0), (1180, 1058, 175))
  canvas.segment_at((8, 19, 976), )
  # runner.run((bbox.start.z, bbox.start.y, bbox.start.x),
  #            (bbox.size.z, bbox.size.y, bbox.size.x))  # Main Body
  counter_path = os.path.join(request.segmentation_output_dir, 'counters.txt')
  if not gfile.Exists(counter_path):
    runner.counters.dump(counter_path)


if __name__ == '__main__':
  app.run(main)
