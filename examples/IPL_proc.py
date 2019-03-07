from analysis_script.utils_format_convert import convert_image_stack_to_h5, normalize_img_stack
from analysis_script.image_preprocess import normalize_img_stack_with_mask
from os.path import join
import numpy as np
from ffn.inference.storage import subvolume_path
from neuroglancer_segment_visualize import neuroglancer_visualize, generate_seg_dict_from_dir, generate_seg_dict_from_dir_list
from ffn.inference.segmentation import relabel_volume
#%% make dataset
path = "/home/morganlab/Documents/ixQ_IPL/"
stack_n = 78
EM_name_pattern = "ixQ_waf009_IPL_export_s%02d.png" # "ixQ_IPL_%03d.png" #
output_name = "grayscale_ixQ_IPL.h5" # "grayscale_ixQ_IPL_align.h5"
EM_stack = convert_image_stack_to_h5(path=path, pattern=EM_name_pattern, stack_n=stack_n, output=output_name)
print("mean: %.2f, std: %.2f" % (EM_stack.mean(), EM_stack.std()))
norm_output_name = output_name[:output_name.find('.h5')] + "_norm.h5"
norm_EM_stack = normalize_img_stack_with_mask(path, norm_output_name, EM_stack,
                              upper=196, lower=80, up_outlier=245, low_outlier=30)
print("mean: %.2f, std: %.2f" % (norm_EM_stack.mean(), norm_EM_stack.std()))
#%%
from neuroglancer_segment_visualize import neuroglancer_visualize
h5_name = join(path, norm_output_name)
neuroglancer_visualize({}, h5_name)
#%% on Cluster inference



#%%
from neuroglancer_segment_visualize import neuroglancer_visualize
h5_name = join(path, "grayscale_ixQ_IPL_align_norm.h5")
seg_dict = generate_seg_dict_from_dir_list(path="/home/morganlab/Documents/ixQ_IPL/",
                                           seg_dir_list=["IPL_exp1-%d"%i for i in range(1,17)])
viewer = neuroglancer_visualize(seg_dict, h5_name)