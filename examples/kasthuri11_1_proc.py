from analysis_script.utils_format_convert import convert_image_stack_to_h5, normalize_img_stack
from analysis_script.image_preprocess import normalize_img_stack_with_mask
from os.path import join
import numpy as np
from ffn.inference.storage import subvolume_path
from neuroglancer_segment_visualize import neuroglancer_visualize, generate_seg_dict_from_dir, generate_seg_dict_from_dir_list
from time import time
path = "/home/morganlab/Documents/Kasthuri11_dataset"
#%% ##############################################################
#%% Import downscaled image to make dataset
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
                                seg_dir_list=["kasthuri_1_exp1-%d"% i for i in range(1,21)])
img_dir = "/home/morganlab/Documents/Kasthuri11_dataset/grayscale_kasthuri_1_norm.h5"
viewer = neuroglancer_visualize(seg_dict, img_dir)
#%% view results
seg_dict = generate_seg_dict_from_dir_list(path="/home/morganlab/Documents/Kasthuri11_dataset/",
                                seg_dir_list=["kasthuri_1_exp1-%d"% i for i in range(41, 55)])
img_dir = "/home/morganlab/Documents/Kasthuri11_dataset/grayscale_kasthuri_1_norm.h5"
viewer = neuroglancer_visualize(seg_dict, img_dir)

#%% view results
seg_dict = generate_seg_dict_from_dir_list(path="/home/morganlab/Documents/Kasthuri11_dataset/",
                                seg_dir_list=["kasthuri_1_exp1_rev-2_tmp/"])
img_dir = "/home/morganlab/Documents/Kasthuri11_dataset/grayscale_kasthuri_1_norm.h5"
viewer = neuroglancer_visualize(seg_dict, img_dir)
#%%
seg_dict = generate_seg_dict_from_dir_list(path="/home/morganlab/Documents/Kasthuri11_dataset/",
                                seg_dir_list=["kasthuri_1_exp1_rev-2/"])
img_dir = "/home/morganlab/Documents/Kasthuri11_dataset/grayscale_kasthuri_1_norm.h5"
viewer = neuroglancer_visualize(seg_dict, img_dir)
#%%
seg_dict = generate_seg_dict_from_dir_list(path="/home/morganlab/Documents/Kasthuri11_dataset/",
                                seg_dir_list=["kasthuri_1_exp1_rev-%d/" % i for i in range(41,69)])
img_dir = "/home/morganlab/Documents/Kasthuri11_dataset/grayscale_kasthuri_1_norm.h5"
viewer = neuroglancer_visualize(seg_dict, img_dir)
#%%
seg_dict = generate_seg_dict_from_dir_list(path="/home/morganlab/Documents/Kasthuri11_dataset/",
                                seg_dir_list=["kasthuri_1_exp1-%d/" % i for i in range(41,56)])
img_dir = "/home/morganlab/Documents/Kasthuri11_dataset/grayscale_kasthuri_1_norm.h5"
viewer = neuroglancer_visualize(seg_dict, img_dir)
#%% ##############################################################
#%% Consensus of 2 stacks
#%% ##############################################################
from run_consensus import run_save_consensus
seg_dict = generate_seg_dict_from_dir("/home/morganlab/Documents/Kasthuri11_dataset/kasthuri_1_exp1_rev-2/")
corners = []
for name, spec in seg_dict.items():
    if type(spec) is dict:
        corners.append(spec['corner'])
config = """
    segmentation1 {
        directory: "/home/morganlab/Documents/Kasthuri11_dataset/kasthuri_1_exp1_rev-2"
        threshold: 0.6
        split_cc: 1
        min_size: 5000
    }
    segmentation2 {
        directory: ???
        threshold: 0.6
        split_cc: 1
        min_size: 5000
    }
    segmentation_output_dir: "/home/morganlab/Documents/Kasthuri11_dataset/kasthuri_1_exp1_rev-2-consensus/"
    type: CONSENSUS_SPLIT
    split_min_size: 5000
    """
cons_seg = run_save_consensus(config, corners=corners)
# Spend 35 mins to consensus on 45 subvolumes
#%%
##  Stitching up the subvolumes!
from analysis_script.subvolume_stitching import stitich_subvolume_grid
t0 = time()
seg_dir = "/home/morganlab/Documents/Kasthuri11_dataset/kasthuri_1_exp1_rev-2/"
full_segment, segment_graph, seg_id_dict = stitich_subvolume_grid(seg_dir, x_step=500, y_step=500, x_num=9, y_num=5, size=(400, 600, 600),
                                                 start_corner=(0, 0, 0), overlap_d=1,
                                                 output_dir="/home/morganlab/Documents/Kasthuri11_dataset/kasthuri_1_exp1_rev-2_full/")
# seems the graph is all correct but the final ourput is not updated!!!
print("Spend ", time()-t0, "s. ")
# Spend  3111.853937149048 s. to stiticha
#%% See the whole volume segmented
seg_dict = generate_seg_dict_from_dir_list(path="/home/morganlab/Documents/Kasthuri11_dataset/",
                                seg_dir_list=["kasthuri_1_exp1_rev-2_full", ])
img_dir = "/home/morganlab/Documents/Kasthuri11_dataset/grayscale_kasthuri_1_norm.h5"
viewer = neuroglancer_visualize(seg_dict, img_dir)

#%% ##############################################################
#%% resegmentation
#%% ##############################################################





#%% ##############################################################
#%% Manual Agglomeration
#%% ##############################################################
#%%###################################################
# Manual agglomeration
######################################################
h5_name = join(path, "grayscale_kasthuri_1_norm.h5")
seg_dict = {'seg': {"seg_dir":"/home/morganlab/Documents/Kasthuri11_dataset/kasthuri_1_exp1_rev-2_full/"}}
viewer = neuroglancer_visualize(seg_dict, h5_name)
#%%
import networkx as nx
import pickle
from analysis_script.neuroglancer_agglomeration import ManualAgglomeration
# seg = np.load("/home/morganlab/Documents/Kasthuri11_dataset/kasthuri_1_exp1_rev-2_full/0/0/seg-0_0_0.npz")
# segmentation = seg["segmentation"]
# seg.close()
# graph = nx.Graph()
# objects = np.unique(segmentation,)
# # assert objects[0]==0
# graph.add_nodes_from(objects[1:])
p = pickle.load(open("/home/morganlab/Documents/Kasthuri11_dataset/kasthuri_1_exp1_rev-2_full/kasthuri_agglomeration.pkl","rb"))
objects, graph = p['objects'], p['graph']
agg_tool = ManualAgglomeration(graph, viewer, objects)
#%%
save_path = "/home/morganlab/Documents/Kasthuri11_dataset/kasthuri_1_exp1_rev-2_full/kasthuri_agglomeration.pkl"
objects, graph = agg_tool.objects, agg_tool.graph
agg_tool.export_merge_data(save_path);


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