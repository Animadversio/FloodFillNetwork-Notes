import numpy as np
import matplotlib.pylab as plt
import neuroglancer
from analysis_script.utils_format_convert import read_image_vol_from_h5
#%%
f = np.load("/home/morganlab/Documents/ixP11LGN/p11_1_exp8/0/0/seg-0_0_0.npz")
v1 = f['segmentation']
f.close()
f = np.load("/home/morganlab/Documents/ixP11LGN/p11_1_exp8/0/448/seg-0_448_0.npz")
v2 = f['segmentation']
f.close()
f = np.load("/home/morganlab/Documents/ixP11LGN/p11_1_exp8/448/0/seg-448_0_0.npz")
v3 = f['segmentation']
f.close()
f = np.load("/home/morganlab/Documents/ixP11LGN/p11_1_exp8/448/448/seg-448_448_0.npz")
v4 = f['segmentation']
f.close()
#%%


plt.imshow(v1[:,-32,:])
plt.show()
plt.imshow(v2[:,32,:])
plt.show()
#%%
BASE = int(v1.max()+1)
composite_map = v1[:,-32,:] + v2[:,32,:] * BASE # note the label width
compo_idx, cnt = np.unique(composite_map, return_counts=True)
#%%
idx2, idx1 = np.divmod(compo_idx, BASE)
#%%
merge_list_2 = []
size_list_2 = []
idx2_set = set(idx2)
for id2 in idx2_set:
    overlap_cnt = cnt[idx2 == id2]
    overlap_label = idx1[idx2 == id2]
    i = overlap_cnt.argmax()
    id1 = overlap_label[i]
    overlap_size = overlap_cnt[i]
    merge_list_2.append((id1, id2))
    size_list_2.append(overlap_size)
#%%
merge_list_1 = []
size_list_1 = []
idx1_set = set(idx1)
for id1 in idx1_set:
    overlap_cnt = cnt[idx1 == id1]
    overlap_label = idx2[idx1 == id1]
    i = overlap_cnt.argmax()
    id2 = overlap_label[i]
    overlap_size = overlap_cnt[i]
    merge_list_1.append((id1, id2))
    size_list_1.append(overlap_size)
#%%
consensus_merge = list(set(merge_list_1) & set(merge_list_2))
consensus_size_list = [(size_list_1[merge_list_1.index(pair)], size_list_2[merge_list_2.index(pair)] ) for pair in consensus_merge]
#%%
threshold = 100 # minimum size for threshold
mask = [1 if (size_pair[1] > threshold & size_pair[0]>threshold) else 0 for size_pair in consensus_size_list]
#%% merge and remap index



#%%
image_stack = read_image_vol_from_h5("/home/morganlab/Documents/ixP11LGN/grayscale_ixP11_1_norm.h5")
#%%
viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    s.voxel_size = [8, 8, 40]
    # s.layers.append(
    #     name='consensus_segmentation',
    #     layer=neuroglancer.LocalVolume(
    #         data=cons_seg,
    #         # offset is in nm, not voxels
    #         # offset=(200, 300, 150),
    #         voxel_size=s.voxel_size,
    #     ),)
    s.layers.append(
        name='seg_exp8',
        layer=neuroglancer.LocalVolume(
            data=v1,
            # offset is in nm, not voxels
            # offset=(200, 300, 150),
            voxel_size=s.voxel_size,
        ), )
    s.layers.append(
        name='seg_exp8-3',
        layer=neuroglancer.LocalVolume(
            data=v2,
            # offset is in nm, not voxels
            offset=(0, 3584, 0),
            voxel_size=s.voxel_size,
        ), )
    s.layers.append(
        name='seg_exp8-2',
        layer=neuroglancer.LocalVolume(
            data=v3,
            # offset is in nm, not voxels
            offset=(3584, 0, 0),
            voxel_size=s.voxel_size,
        ), )
    s.layers.append(
        name='seg_exp8-4',
        layer=neuroglancer.LocalVolume(
            data=v4,
            # offset is in nm, not voxels
            offset=(3584, 3584, 0),
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
#     s.layers.append(
#         name='b', layer=neuroglancer.LocalVolume(
#             data=v2,
#             voxel_size=s.voxel_size,
#         ))

print(viewer)
