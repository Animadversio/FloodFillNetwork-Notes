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
logging.getLogger().setLevel(logging.INFO) # set the information level to show INFO logs
# reload(inference)
reload(resegmentation_analysis)

#%%
#"/tmp/LR_model/model.ckpt-3680"
config = """inference {
    image {
      hdf5: "/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/grayscale_maps_LR.h5:raw"
    }
    image_mean: 128
    image_stddev: 33
    checkpoint_interval: 1800
    seed_policy: "PolicyPeaks"
    model_checkpoint_path: "/home/morganlab/Downloads/ffn-master/models/LR_model2/model.ckpt-10000"
    model_name: "convstack_3d.ConvStack3DFFNModel"
    model_args: "{\\"depth\\": 9, \\"fov_size\\": [55, 37, 17], \\"deltas\\": [9,6,3]}"
    segmentation_output_dir: "/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR"
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
output_directory: "/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR/reseg"
max_retry_iters: 2
segment_recovery_fraction: 0.4
analysis_radius {x: 200 y: 200 z: 20}
"""
#%%
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
reseg_req = inference_pb2.ResegmentationRequest()
_ = text_format.Parse(config, reseg_req)
req = reseg_req.inference
# req = inference_pb2.InferenceRequest()
# _ = text_format.Parse(config, req)
#%%
runner = inference.Runner()
runner.start(req)
runner.set_pixelsize([8, 12, 30])

#%%
# seg_canvas1 = runner.run((0, 0, 0), (1000, 1000, 175))
resegmentation.process(reseg_req,runner)
#%%
import numpy as np
from analysis_script.utils_format_convert import read_image_vol_from_h5
import neuroglancer
viewer = neuroglancer.Viewer()
f=np.load("/home/morganlab/Downloads/ffn-master/results/LGN/testing_exp7/0/0/seg-0_0_0.npz")
seg = f['segmentation']
f.close()

#%%
result_proto = resegmentation_analysis.evaluate_pair_resegmentation("/home/morganlab/Documents/Autoseg_result/Autoseg_exp7/reseg/120-1279_at_672_582_92.npz",
                                                                    seg, [20, 200, 200], [20, 200, 200], sampling=[30, 12, 8])
#%%
result_proto2 = resegmentation_analysis.evaluate_pair_resegmentation("/home/morganlab/Documents/Autoseg_result/Autoseg_exp7/reseg/1279-1235_at_689_564_92.npz",
                                                                    seg, [20, 200, 200], [20, 200, 200], sampling=[30, 12, 8])


#%% In situ visualization
image_stack = read_image_vol_from_h5("/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/grayscale_maps_LR(copy).h5")
#%%
with viewer.txn() as s:
    s.voxel_size = [8, 12, 30]
    s.layers.append(
        name='seg_exp7',
        layer=neuroglancer.LocalVolume(
            data=seg,
            # offset is in nm, not voxels
            # offset=(200, 300, 150),
            voxel_size=s.voxel_size,
        ), )
    s.layers.append(
        name='EM_image',
        layer=neuroglancer.LocalVolume(
            data=image_stack,
            voxel_size=s.voxel_size,
        ), )
#         shader="""
# void main() {
#   emitRGB(vec3(toNormalized(getDataValue(0)),
#                toNormalized(getDataValue(1)),
#                toNormalized(getDataValue(2))));
# }
# """)

print(viewer)