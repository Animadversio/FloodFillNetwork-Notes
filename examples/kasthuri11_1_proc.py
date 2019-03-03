from analysis_script.utils_format_convert import convert_image_stack_to_h5, normalize_img_stack
from analysis_script.image_preprocess import normalize_img_stack_with_mask
from os.path import join
import numpy as np
from ffn.inference.storage import subvolume_path
from neuroglancer_segment_visualize import neuroglancer_visualize, generate_seg_dict_from_dir, generate_seg_dict_from_dir_list

#%% ##############################################################
#%% Import downscaled image to make dataset
path = "/home/morganlab/Documents/Kasthuri11_dataset"
stack_n = 400
EM_name_pattern = "Kasthuri11_dataset_1_%03d.png"
output_name = "grayscale_kasthuri_1.h5"
EM_stack = convert_image_stack_to_h5(path=path, pattern=EM_name_pattern, stack_n=stack_n, beg_n=500, output=output_name)
print("mean: %.2f, std: %.2f" % (EM_stack.mean(), EM_stack.std()))
norm_output_name = output_name[:output_name.find('.h5')] + "_norm.h5"
norm_EM_stack = normalize_img_stack_with_mask(path, norm_output_name, EM_stack,
                              upper=196, lower=80, up_outlier=245, low_outlier=30)
print("mean: %.2f, std: %.2f" % (norm_EM_stack.mean(), norm_EM_stack.std()))
#%%
del EM_stack,norm_EM_stack
#%% Check image stack
from neuroglancer_segment_visualize import neuroglancer_visualize
h5_name = join(path, norm_output_name)
neuroglancer_visualize({}, "/home/morganlab/Documents/Kasthuri11_dataset/grayscale_kasthuri_1_norm.h5")
#%% ##############################################################
#%% Make masks for tissue types
#%%
#%% ##############################################################
#%% Do inference online


# Exp1 screen through the models on line

# Exp0 pick some 1-2 models and do large scale segmentation

import glob


#%% ##############################################################
#%% view results
seg_dict = generate_seg_dict_from_dir_list(path="/home/morganlab/Documents/Kasthuri11_dataset/",
                                seg_dir_list=["kasthuri_1_exp1-%d"%i for i in range(1,21)])
img_dir = "/home/morganlab/Documents/Kasthuri11_dataset/grayscale_kasthuri_1_norm.h5"
viewer = neuroglancer_visualize(seg_dict, img_dir)
#%%
seg_dict = generate_seg_dict_from_dir_list(path="/home/morganlab/Documents/ixP11LGN/",
                                seg_dir_list=["p11_6_exp1-1", "p11_6_exp1-2", "p11_6_exp1-21", "p11_6_exp1-22",
                                              "p11_6_exp1-23", "p11_6_exp1-24", "p11_6_exp1-25", "p11_6_exp1-26",
                                              "p11_6_exp1-27", "p11_6_exp1-28", "p11_6_exp1-29", "p11_6_exp1-30", ])
img_dir = "/home/morganlab/Documents/ixP11LGN/EM_data/p11_6_EM/grayscale_ixP11_6_align_norm.h5"
viewer = neuroglancer_visualize(seg_dict, img_dir)
#%%
seg_dict = generate_seg_dict_from_dir("/home/morganlab/Documents/ixP11LGN/p11_6_exp1-38")
img_dir = "/home/morganlab/Documents/ixP11LGN/EM_data/p11_6_EM/grayscale_ixP11_6_align_norm.h5"
viewer2 = neuroglancer_visualize(seg_dict, img_dir)
#%%
seg_dict = generate_seg_dict_from_dir("/home/morganlab/Documents/ixP11LGN/p11_6_exp1-33")
img_dir = "/home/morganlab/Documents/ixP11LGN/EM_data/p11_6_EM/grayscale_ixP11_6_align_norm.h5"
viewer2 = neuroglancer_visualize(seg_dict, img_dir)
#%%
seg_dict = generate_seg_dict_from_dir_list(path="/home/morganlab/Documents/ixP11LGN/",
                                seg_dir_list=["p11_6_exp1-33", "p11_6_exp1-38",])
img_dir = "/home/morganlab/Documents/ixP11LGN/EM_data/p11_6_EM/grayscale_ixP11_6_align_norm.h5"
viewer = neuroglancer_visualize(seg_dict, img_dir)
#%% ##############################################################
#%% Consensus of 2 stacks
#%% ##############################################################
from run_consensus import run_save_consensus
from time import time
from analysis_script.subvolume_stitching import stitich_subvolume_grid
seg_dict = generate_seg_dict_from_dir("/home/morganlab/Documents/ixP11LGN/p11_6_exp1-38")
corners = []
for name, spec in seg_dict.items():
    if type(spec) is dict:
        corners.append(spec['corner'])
config = """
    segmentation1 {
        directory: "/home/morganlab/Documents/ixP11LGN/p11_6_exp1-38/"
        threshold: 0.6
        split_cc: 1
        min_size: 5000
    }
    segmentation2 {
        directory: "/home/morganlab/Documents/ixP11LGN/p11_6_exp1-33/"
        threshold: 0.6
        split_cc: 1
        min_size: 5000
    }
    segmentation_output_dir: "/home/morganlab/Documents/ixP11LGN/p11_6_consensus_33_38/"
    type: CONSENSUS_SPLIT
    split_min_size: 5000
    """
cons_seg = run_save_consensus(config, corners=corners)
# Spend 35 mins to consensus on 45 subvolumes

##  Stitching up the subvolumes!
t0 = time()
seg_dir = "/home/morganlab/Documents/ixP11LGN/p11_6_consensus_33_38/"
full_segment, segment_graph, seg_id_dict = stitich_subvolume_grid(seg_dir, x_step=500, y_step=500, x_num=9, y_num=5, size=(152, 600, 600),
                                                 start_corner=(0, 0, 0), overlap_d=1,
                                                 output_dir="/home/morganlab/Documents/ixP11LGN/p11_6_consensus_33_38_full")
# seems the graph is all correct but the final ourput is not updated!!!
print("Spend ", time()-t0, "s. ")
# Spend  1168.8590106964111 s.  for stitching up 45 subvolumes
#%% See the whole volume segmented
seg_dict = generate_seg_dict_from_dir_list(path="/home/morganlab/Documents/ixP11LGN/",
                                seg_dir_list=["p11_6_consensus_33_38_full", ])
img_dir = "/home/morganlab/Documents/ixP11LGN/EM_data/p11_6_EM/grayscale_ixP11_6_align_norm.h5"
viewer = neuroglancer_visualize(seg_dict, img_dir)

#%% ##############################################################
#%% resegmentation
#%% ##############################################################





#%% ##############################################################
#%% Manual Agglomeration
#%% ##############################################################
from ffn.utils.proofreading import GraphUpdater, ObjectReview
import networkx as nx
from neuroglancer_segment_visualize import GraphUpdater_show

# class GraphUpdater_show(GraphUpdater):
#     def set_init_state(self):
#         self.viewer = neuroglancer_visualize(self.seg_dict, self.img_dir)
#
#     def __init__(self, graph, objects, bad, seg_dict, img_dir):
#         self.seg_dict = seg_dict
#         self.img_dir = img_dir
#         super(GraphUpdater_show, self).__init__(graph, objects, bad)

graph = nx.Graph()
seg = np.load(subvolume_path("/home/morganlab/Documents/ixP11LGN/p11_6_consensus_33_38_full/", (0, 0, 0), "npz"))
segmentation = seg["segmentation"]
seg.close()
objects, cnts = np.unique(segmentation, return_counts=True)
objects = objects
image_dir = "/home/morganlab/Documents/ixP11LGN/EM_data/grayscale_ixP11_5_align_norm_new.h5"
graph_update = GraphUpdater_show(graph, objects, [], {'seg':{"vol":segmentation}, }, image_dir)
#%%

#%%
image_size = (152, 4474, 2383)
full_segment[:, :, 4474:] = 0
full_segment[:, 2383:, :] = 0

#%%##############################################################
#%% Output to KNOSSOS to do manual merging
#%%##############################################################
import knossos_utils
import numpy as np
img_dir = "/home/morganlab/Documents/ixP11LGN/EM_data/p11_6_EM/grayscale_ixP11_6_align_norm.h5"
kns_dataset = knossos_utils.KnossosDataset()
kns_dataset.initialize_from_matrix(path="/home/morganlab/Documents/ixP11LGN/p11_6_Knossos_dataset/",
                                   scale=(40, 8, 8),
                                   experiment_name="p11_6_LGN",
                                   data_path=img_dir,  hdf5_names=["raw"],
                                   mags=[1, 2, 4, 6, 8])
#%%
# from knossos_utils.knossosdataset import  load_from_h5py
# data = load_from_h5py(img_dir, hdf5_names="raw")
#%%
# import h5py
# f = h5py.File(img_dir,'r')
# img_stack = f['raw'].value
# f.close()
#%% Export the cube dataset into KNOSSOS
kzip_path = "/home/morganlab/Documents/ixP11LGN/p11_6_Knossos_dataset/annotation/consensus_33_38_full" # note
kns_dataset.from_matrix_to_cubes((0,0,0), mags=[1, 2, 4, 6, 8], data=full_segment, data_mag=1,
                             data_path=None, hdf5_names=None,
                             datatype=np.uint64, fast_downsampling=True, kzip_path=kzip_path)