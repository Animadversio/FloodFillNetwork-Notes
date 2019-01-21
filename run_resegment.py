"""
Run Resegmentation for a
"""

import numpy as np
# import matplotlib.pylab as plt

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
from importlib import reload
import logging
# reload(inference)
# reload(resegmentation)
logging.getLogger().setLevel(logging.INFO) # set the information level to show INFO logs
#%%
#"/tmp/LR_model/model.ckpt-3680"
config = """inference {
    image {
      hdf5: "/Users/binxu/Connectomics_Code/LGN_DATA/grayscale_maps_LR2.h5:raw"
    }
    image_mean: 128
    image_stddev: 33
    checkpoint_interval: 1800
    seed_policy: "PolicyPeaks"
    model_checkpoint_path: "/Users/binxu/Connectomics_Code/LR_model_WF/model.ckpt-10000"
    model_name: "convstack_3d.ConvStack3DFFNModel"
    model_args: "{\\"depth\\": 9, \\"fov_size\\": [55, 37, 17], \\"deltas\\": [9,6,3]}"
    segmentation_output_dir: "/Users/binxu/Connectomics_Code/results/LGN/testing_LR"
    inference_options {
      init_activation: 0.95
      pad_value: 0.05
      move_threshold: 0.9
      min_boundary_dist { x: 3 y: 3 z: 1}
      segment_threshold: 0.6
      min_segment_size: 1000
    }
    init_segmentation {
       npz: "/Users/binxu/Connectomics_Code/results/LGN/testing_LR/0/0/seg-0_0_0.npz:segmentation"
    }
}
points {id_a:22 id_b:29 point {x: 158 y: 510 z: 8} }
points {id_a:432 id_b:636 point {x: 334 y: 542 z: 32} }
radius {x: 200 y: 200 z: 20}
output_directory: "/Users/binxu/Connectomics_Code/results/LGN/testing_LR/reseg"
max_retry_iters: 2
segment_recovery_fraction: 0.4
analysis_radius {x: 200 y: 200 z: 20}
"""

reseg_req = inference_pb2.ResegmentationRequest()
_ = text_format.Parse(config, reseg_req)
req = reseg_req.inference
# req = inference_pb2.InferenceRequest()
# _ = text_format.Parse(config, req)
#%%
runner = inference.Runner()
runner.start(req)
runner.set_pixelsize([8,12,30])

#%%
# seg_canvas1 = runner.run((0, 0, 0), (1000, 1000, 175))
resegmentation.process(reseg_req, runner)