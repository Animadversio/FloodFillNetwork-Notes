from analysis_script.utils_format_convert import convert_image_stack_to_h5, normalize_img_stack
from analysis_script.image_preprocess import normalize_img_stack_with_mask
from os.path import join
import numpy as np
from ffn.inference.storage import subvolume_path
from time import time
from neuroglancer_segment_visualize import neuroglancer_visualize, generate_seg_dict_from_dir, generate_seg_dict_from_dir_list
from ffn.inference.segmentation import relabel_volume
path = "/home/morganlab/Documents/ixQ_IPL/"
#%% Add training dataset
from analysis_script.utils_format_convert import convert_raw_seg_stack_to_h5
path = "/home/morganlab/Documents/ixQ_IPL"
# pattern = "Segmentation1-LX_8-14.vsseg_export_s%03d.png"
raw_pattern = "KV_IxQ_retinaGroundTruth_2019+23+4.vsseg_export_s%02d_5152x4848_8bpp.raw"
raw_shape = (4839, 5146,)
img_shape = (4839, 5146,)
stack_n = 78
seg_stacks = convert_raw_seg_stack_to_h5(path, raw_pattern, stack_n, raw_shape, img_shape,
                             beg_n=0, output="IxQ_retina_groundtruth.h5", in_dtype=np.int8, out_dtype='int16') #FIXME!!!! dtype
# (5146, 4839)
# KV_IxQ_retinaGroundTruth_2019+23+4.vsseg_export

#%%
seg_stacks_enhance = np.uint16(seg_stacks)
viewer = neuroglancer_visualize({"ground_truth":{"vol":seg_stacks_enhance}}, h5_name)
#%% Correct manual segmentation split
seg_stacks_enhance[seg_stacks_enhance == 100] = 97
#%%
import h5py
f = h5py.File(join(path, "IxQ_retina_groundtruth.h5"), "w")
fstack=f.create_dataset("stack", (stack_n, *img_shape,), dtype="uint16") # Note they only take int64 input
fstack[:] = seg_stacks_enhance
#%%%
import matplotlib.pylab as plt
plt.figure()
plt.imshow(seg_stacks[1,:,:])
plt.show()
#%%


#%% make h5 dataset
path = "/home/morganlab/Documents/ixQ_IPL/"
stack_n = 78
EM_name_pattern = "ixQ_waf009_IPL_export_s%02d.png" # "ixQ_IPL_%03d.png" #
output_name = "grayscale_ixQ_IPL_tmp.h5" # "grayscale_ixQ_IPL_align.h5"
EM_stack = convert_image_stack_to_h5(path=path, pattern=EM_name_pattern, stack_n=stack_n, output=output_name)
print("mean: %.2f, std: %.2f" % (EM_stack.mean(), EM_stack.std()))
norm_output_name = output_name[:output_name.find('.h5')] + "_norm.h5"
norm_EM_stack = normalize_img_stack_with_mask(path, norm_output_name, EM_stack,
                              upper=196, lower=80, up_outlier=245, low_outlier=30)
# Normalize image with mask (masking out the outlier pixels)
print("mean: %.2f, std: %.2f" % (norm_EM_stack.mean(), norm_EM_stack.std()))
#%% Preliminary check of h5 image dataset (normalization & registration)
h5_name = join(path, "grayscale_ixQ_IPL_align_norm.h5") # norm_output_name)
neuroglancer_visualize({}, h5_name)
#%% on Cluster inference



#%%
h5_name = join(path, "grayscale_ixQ_IPL_align_norm.h5")
seg_dict = generate_seg_dict_from_dir_list(path="/home/morganlab/Documents/ixQ_IPL/",
                                           seg_dir_list=["IPL_exp1-%d"%i for i in range(3,17)])
viewer = neuroglancer_visualize(seg_dict, h5_name)
#%%
h5_name = join(path, "grayscale_ixQ_IPL_align_norm.h5")
seg_dict = generate_seg_dict_from_dir_list(path="/home/morganlab/Documents/ixQ_IPL/",
                                           seg_dir_list=["IPL_exp1-%d"%i for i in range(17,42)])
viewer = neuroglancer_visualize(seg_dict, h5_name)
#%%
#h5_name = join(path, "grayscale_ixQ_IPL_align_norm.h5")
seg_dict = generate_seg_dict_from_dir_list(path="/Users/binxu/Connectomics_Code/results/IPL/",
                                           seg_dir_list=["IPL_exp1-%d"%i for i in range(10,16)])
viewer = neuroglancer_visualize(seg_dict, None)
#%%
h5_name = join(path, "grayscale_ixQ_IPL_align_norm.h5")
seg_dict = generate_seg_dict_from_dir_list(path="/home/morganlab/Documents/ixQ_IPL/",
                                           seg_dir_list=["IPL_exp1-2", "IPL_exp1-2_rev"])
viewer = neuroglancer_visualize(seg_dict, h5_name)
#%%
#%%
from run_consensus import run_save_consensus
seg_dict = generate_seg_dict_from_dir("/home/morganlab/Documents/ixQ_IPL/IPL_exp1-2_rev/")
corners = []
for name, spec in seg_dict.items():
    if type(spec) is dict:
        corners.append(spec['corner'])
config = """
    segmentation1 {
        directory: "/home/morganlab/Documents/ixQ_IPL/IPL_exp1-2/"
        threshold: 0.6
        split_cc: 1
        min_size: 5000
    }
    segmentation2 {
        directory: "/home/morganlab/Documents/ixQ_IPL/IPL_exp1-2_rev/"
        threshold: 0.6
        split_cc: 1
        min_size: 5000
    }
    segmentation_output_dir: "/home/morganlab/Documents/ixQ_IPL/IPL_exp1-2_rev_consensus/"
    type: CONSENSUS_SPLIT
    split_min_size: 5000
    """
cons_seg = run_save_consensus(config, corners=corners)
# Spent 32min to do 25 consensus
#%%
from analysis_script.subvolume_stitching import stitich_subvolume_grid
t0 = time()
seg_dir = "/home/morganlab/Documents/ixQ_IPL/IPL_exp1-2_rev_consensus/"
full_segment, segment_graph, seg_id_dict = stitich_subvolume_grid(seg_dir, x_step=1000, y_step=1000, x_num=5, y_num=5, size=(78, 1100, 1100),
                                                 start_corner=(0, 0, 0), overlap_d=1,
                                                 output_dir="/home/morganlab/Documents/ixQ_IPL/IPL_exp1-2_rev_consensus_full/")
# seems the graph is all correct but the final ourput is not updated!!!
print("Spend ", time()-t0, "s. ")
# Spend  864.5802557468414 s.

#%%
h5_name = join(path, "grayscale_ixQ_IPL_align_norm.h5")
seg_dict = generate_seg_dict_from_dir_list(path="/home/morganlab/Documents/ixQ_IPL/",
                                           seg_dir_list=["IPL_exp1-2_rev_consensus_full/",])
viewer = neuroglancer_visualize(seg_dict, h5_name)
#%%###################################################
# Manual agglomeration
######################################################
h5_name = join(path, "grayscale_ixQ_IPL_align_norm.h5")
seg_dict = {'seg': {"seg_dir":"/home/morganlab/Documents/ixQ_IPL/IPL_exp1-2_rev_consensus_full/"}}
    # generate_seg_dict_from_dir_list(path="/home/morganlab/Documents/ixQ_IPL/",
    #                                        seg_dir_list=["IPL_exp1-2_rev_consensus_full/",])
viewer = neuroglancer_visualize(seg_dict, h5_name)
# %%
import networkx as nx
import pickle
from analysis_script.neuroglancer_agglomeration import ManualAgglomeration
seg = np.load("/home/morganlab/Documents/ixQ_IPL/IPL_exp1-2_rev_consensus/0/0/seg-0_0_0.npz")
segmentation = seg["segmentation"]
seg.close()
graph = nx.Graph()
objects = np.unique(segmentation,)
assert objects[0]==0
graph.add_nodes_from(objects[1:])
agg_tool = ManualAgglomeration(graph, viewer, )
# %%
save_path = "/home/morganlab/Documents/ixQ_IPL/IPL_exp1-2_rev_consensus/IPL_agglomeration.pkl"
objects, graph = agg_tool.objects, agg_tool.graph
agg_tool.export_merge_data(save_path);