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
from ffn.inference import seed

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
    }
}
points {id_a:22 id_b:29 point {x: 510 y: 158 z: 8} } 
points {id_a:432 id_b:636 point {x: 550 y: 331 z: 32} } 
radius {x: 200 y: 200 z: 20}
output_directory: "/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR/reseg"
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
#%% Recover the canvas but it does not recover the segmentation!
canvas, alignment = runner.make_canvas((0, 0, 0), (175, 1000, 1000))

#%%
ResultPath = "/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR/0/0/"
testSegLoc = ResultPath + "seg-0_0_0.npz"
testProbLoc = ResultPath + 'seg-0_0_0.prob'
data = np.load(testSegLoc)
segmentation = data['segmentation']
#%%
seg = segmentation.reshape((1, *segmentation.shape))
canvas.init_segmentation_from_volume(seg,(0,0,0),(175, 1000, 1000))
#%%
# seg_canvas1 = runner.run((0, 0, 0), (1000, 1000, 175))
resegmentation.process(reseg_req,runner)


#%% Seed policy tuning !
seed_policy = seed.PolicyPeaks(canvas)
seed_policy._init_coords()
seed_list = seed_policy.coords
#%%
import pickle
pickle.dump(seed_list, open("/home/morganlab/Downloads/LGN_Autoseg_Result/Seed_Distribution/seed_list.pkl",'wb'))
#%%
import h5py
f = h5py.File("/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/grayscale_maps_LR.h5",'r')

from ffn.inference import storage
volume = storage.decorated_volume(req.image)
#%%
volume = volume.value.copy()
#%%%
for zid in range(175):
    seeds_in_layer = seed_list[seed_list[:,0]==zid]
    plt.imshow(volume[zid,:,:],cmap='gray')
    plt.scatter(seeds_in_layer[:,2],seeds_in_layer[:,1],c='red',s=0.5) # note the way they scatter, Xaxis corr to 3rd index
    plt.axis('off')
    plt.savefig('seed_distribution_%d.png'%zid)
    plt.show()
    plt.close()

