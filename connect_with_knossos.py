"""
Create KNOSSOS dataset from the image data and segmentation files for proofreading
and post processing

"""

import knossos_utils
from ffn.inference.storage import subvolume_path
import numpy as np
#%%
kns_dataset = knossos_utils.KnossosDataset()
kns_dataset.initialize_from_knossos_path("/home/morganlab/Documents/ixP11LGN/Knossos_dataset/",)
#%%
# kns_dataset.initialize_without_conf("/home/morganlab/Documents/ixP11LGN/Knossos_dataset",
#                            boundary=size,
#                            scale=resolution,
#                            experiment_name="p11_5_LGN")
#%%
import h5py
image_dir = "/home/morganlab/Documents/ixP11LGN/EM_data/grayscale_ixP11_5_align_norm_new.h5"
fmap = h5py.File(image_dir, 'r')
img_stack = fmap['raw']
img_stack = img_stack[:, :, :]
fmap.close()
#%%
# viewer = neuroglancer_visualize({}, image_dir)

#%% Create dataset with loaded nparray dataset@!
# Note path must be ended with /
kns_dataset.initialize_from_matrix(path="/home/morganlab/Documents/ixP11LGN/Knossos_dataset/",
                                   scale=(40, 8, 8),
                                   experiment_name="p11_5_LGN",
                                   data=img_stack, mags=[1, 2, 4, 8])
#%% (Alternatively) Create dataset from h5py file
kns_dataset.initialize_from_matrix(path="/home/morganlab/Documents/ixP11LGN/Knossos_dataset/",
                                   scale=(40, 8, 8),
                                   experiment_name="p11_5_LGN",
                                   data_path="/home/morganlab/Documents/ixP11LGN/EM_data/grayscale_ixP11_5_align_norm_new.h5",
                                   mags=[1, 2, 4, 8])

#%% Fetch segmentation file from npz
corner = (0, 1000, 1500)
seg_dir = "/home/morganlab/Documents/ixP11LGN/p11_5_exp1_rev_full"
data_path = subvolume_path(seg_dir, corner, 'npz')
seg = np.load(data_path)
seg=seg['segmentation']
#%% Export the cube dataset into KNOSSOS
kzip_path = "/home/morganlab/Documents/ixP11LGN/Knossos_dataset/annotation/Autoseg" # note
kns_dataset.from_matrix_to_cubes(corner, mags=[1, 2, 4, 8], data=seg, data_mag=1,
                             data_path=None, hdf5_names=None,
                             datatype=np.uint64, fast_downsampling=True, kzip_path=kzip_path)