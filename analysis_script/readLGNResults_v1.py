# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#%%
testDatLoc='/home/morganlab/Downloads/ffn-master/results/fib25/sample-training2.npz'
data=np.load(testDatLoc)
dataFast=np.load(testDatLoc, mmap_mode='r')

segs=data['segmentation']
testimage=segs[50,:,:]
testimagebit=testimage*7
img = Image.fromarray(testimagebit)
img
#%% 
testSegLoc='/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR/0/0/seg-0_0_0.npz'
testProbLoc='/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR/0/0/seg-0_0_0.prob'
data=np.load(testSegLoc)
segmentation = data['segmentation']
#%%
data_prob = np.load(testProbLoc)
qprob = data_prob['qprob']
#%%
plt.figure()
plt.imshow(segmentation[50,:,:])
plt.show()
#%%
idx,cnts=np.unique(segmentation,return_counts=True)
#%%
idx,cnts=np.unique(segmentation[125,:,:],return_counts=True)
#%%
plt.hist(np.log10(cnts),bins=50)
plt.show()

#%%

cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
#%%
exportLoc = '/home/morganlab/Downloads/LGN_Autoseg_Result/'
for i in range(segmentation.shape[0]):
    plt.figure()
    # plt.imshow(segmentation[i,:,:])
    plt.imsave(exportLoc+"seg_%03d.tif"%i,segmentation[i,:,:])
    plt.close()
#%%
tmp = plt.imread(exportLoc+"seg_%03d.tif"%10)
#%%
from google.protobuf import text_format
from ffn.inference import inference
from ffn.inference import inference_pb2
#%%
runner = inference.Runner()
#%%
config = '''image {
  hdf5: "third_party/LGN_DATA/grayscale_maps_zyx.h5:raw"
}
image_mean: 128
image_stddev: 33
checkpoint_interval: 1800
seed_policy: "PolicyPeaks"
model_checkpoint_path: "/tmp/big_model3/model.ckpt-3136"
model_name: "convstack_3d.ConvStack3DFFNModel"
model_args: "{\\"depth\\": 12, \\"fov_size\\": [49, 49, 49], \\"deltas\\": [0, 0, 0]}"
segmentation_output_dir: "results/fib25/training2"
inference_options {
  init_activation: 0.95
  pad_value: 0.05
  move_threshold: 0.9
  min_boundary_dist { x: 1 y: 1 z: 1}
  segment_threshold: 0.6
  min_segment_size: 1000
}'''
req = inference_pb2.InferenceRequest()
_ = text_format.Parse(config, req)
#%%
runner.start(req)
