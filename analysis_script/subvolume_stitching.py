import numpy as np
import matplotlib.pylab as plt
import neuroglancer
from analysis_script.utils_format_convert import read_image_vol_from_h5
from ffn.inference.storage import subvolume_path
from neuroglancer_segment_visualize import neuroglancer_visualize
from run_consensus import run_save_consensus
import networkx
import ffn.inference.storage as storage
#%%
# seg_dir = "/home/morganlab/Documents/ixP11LGN/p11_1_exp10" # "/Users/binxu/Connectomics_Code/results/LGN/p11_1_exp8" # "/home/morganlab/Documents/ixP11LGN/p11_1_exp8"
# f = np.load(subvolume_path(seg_dir, (0, 0, 0), 'npz'))
# v1 = f['segmentation']
# f.close()
# f = np.load(subvolume_path(seg_dir, (0, 448, 0), 'npz'))
# v2 = f['segmentation']
# f.close()
# f = np.load(subvolume_path(seg_dir, (0, 0, 448), 'npz'))
# v3 = f['segmentation']
# f.close()
# f = np.load(subvolume_path(seg_dir, (0, 448, 448), 'npz'))
# v4 = f['segmentation']
# f.close()
#
# #%%
# seg_dict = {"seg_dir": "/home/morganlab/Documents/ixP11LGN/p11_1_exp10",
#             "seg_1": {"corner": (0, 0, 0)},
#             "seg_2": {"corner": (0, 0, 448)},
#             "seg_3": {"corner": (0, 448, 0)},
#             "seg_4": {"corner": (0, 448, 448)}
#             }
# image_dir = "/home/morganlab/Documents/ixP11LGN/grayscale_ixP11_1_norm.h5"
# neuroglancer_visualize(seg_dict, "/home/morganlab/Documents/ixP11LGN/grayscale_ixP11_1_norm.h5")
# #%%
# config = """
#     segmentation1 {
#         directory: "/home/morganlab/Documents/ixP11LGN/p11_1_exp10"
#         threshold: 0.6
#         split_cc: 1
#         min_size: 5000
#     }
#     segmentation2 {
#         directory: "/home/morganlab/Documents/ixP11LGN/p11_1_exp10_rev"
#         threshold: 0.6
#         split_cc: 1
#         min_size: 5000
#     }
#     segmentation_output_dir: "/home/morganlab/Documents/ixP11LGN/p11_1_exp10_consensus_rev/"
#     type: CONSENSUS_SPLIT
#     split_min_size: 5000
#     """
# run_save_consensus(config, [(0, 0, 0), (0, 0, 448), (0, 448, 0), (0, 448, 448)])
# #%%
# seg_dict = {"seg_dir": "/home/morganlab/Documents/ixP11LGN/p11_1_exp10_consensus_rev/",
#             "seg_1": {"corner": (0, 0, 0)},
#             "seg_2": {"corner": (0, 0, 448)},
#             "seg_3": {"corner": (0, 448, 0)},
#             "seg_4": {"corner": (0, 448, 448)}
#             }
# image_dir = "/home/morganlab/Documents/ixP11LGN/grayscale_ixP11_1_norm.h5"
# neuroglancer_visualize(seg_dict, "/home/morganlab/Documents/ixP11LGN/grayscale_ixP11_1_norm.h5")
# #%%
#
# plt.imshow(v1[:,-33:-30,:])
# plt.show()
# plt.imshow(v2[:,31:34,:])
# plt.show()
# #%% volume slicing function
# corner1 = (0, 0, 0)
# corner2 = (0, 448, 0)
# size = (152, 512, 512)
# size2 = (152, 512, 512)
# overlap_d = 3
def _overlap_selection(corner1, corner2, size, size2=None, overlap_d=3):
    '''Return the middle of overlap subvolume to do next overlap analysis
    :return : sel1 sel2 2 slice object that can send into v1 v2
    '''
    if size2==None:
        size2 = size
    if corner1[0] == corner2[0] and corner1[2] == corner2[2]:  # junction in y axis
        if corner2[1] > corner1[1] and corner1[1] + size[1] > corner2[1]:
            assert ( corner1[1] + size[1] - corner2[1] )%2 == 0
            halfwid = ( corner1[1] + size[1] - corner2[1] )//2
            sel1 = (slice(None), slice(-halfwid - overlap_d, -halfwid + overlap_d), slice(None))
            sel2 = (slice(None), slice(halfwid - overlap_d, halfwid + overlap_d), slice(None))
        elif corner1[1] > corner2[1] and corner2[1] + size[1] > corner1[1]:
            assert (corner2[1] + size[1] - corner1[1]) % 2 == 0
            halfwid = (corner2[1] + size[1] - corner1[1]) // 2
            sel1 = (slice(None), slice(halfwid - overlap_d, halfwid + overlap_d), slice(None))
            sel2 = (slice(None), slice(-halfwid - overlap_d, -halfwid + overlap_d), slice(None))
        else:
            return ([],[],[]), ([],[],[])
    elif corner1[0] == corner2[0] and corner1[1] == corner2[1]:  # junction in x axis
        if corner2[2] > corner1[2] and corner1[2] + size[2] > corner2[2]:
            assert ( corner1[2] + size[2] - corner2[2] )%2 == 0
            halfwid = ( corner1[2] + size[2] - corner2[2] )//2
            sel1 = (slice(None), slice(None), slice(-halfwid - overlap_d, -halfwid + overlap_d))
            sel2 = (slice(None), slice(None), slice(halfwid - overlap_d, halfwid + overlap_d))
        elif corner1[2] > corner2[2] and corner2[2] + size[2] > corner1[2]:
            assert (corner2[2] + size[2] - corner1[2]) % 2 == 0
            halfwid = (corner2[2] + size[2] - corner1[2]) // 2
            sel1 = (slice(None), slice(None), slice(halfwid - overlap_d, halfwid + overlap_d))
            sel2 = (slice(None), slice(None), slice(-halfwid - overlap_d, -halfwid + overlap_d))
        else:
            return ([],[],[]), ([],[],[])
    elif corner1[1] == corner2[1] and corner1[2] == corner2[2]:  # junction in z axis
        if corner2[0] > corner1[0] and corner1[0] + size[0] > corner2[0]:
            assert ( corner1[0] + size[0] - corner2[0] )%2 == 0
            halfwid = ( corner1[0] + size[0] - corner2[0] )//2
            sel1 = (slice(-halfwid - overlap_d, -halfwid + overlap_d), slice(None), slice(None))
            sel2 = (slice(halfwid - overlap_d, halfwid + overlap_d), slice(None), slice(None))
        elif corner1[0] > corner2[0] and corner2[0] + size[0] > corner1[0]:
            assert (corner2[0] + size[0] - corner1[0]) % 2 == 0
            halfwid = (corner2[0] + size[0] - corner1[0]) // 2
            sel1 = (slice(halfwid - overlap_d, halfwid + overlap_d), slice(None), slice(None))
            sel2 = (slice(-halfwid - overlap_d, -halfwid + overlap_d), slice(None), slice(None))
        else:
            return ([],[],[]), ([],[],[])
    else:
        return ([],[],[]), ([],[],[])
    return sel1, sel2
#%%
def merge_segment(v1, v2, corner1, corner2, size, size2=None, overlap_d=3, threshold=100):
    v1 = np.uint64(v1)
    v2 = np.uint64(v2) # note without enough byte space the coposite map method will fail
    sel1, sel2 = _overlap_selection(corner1, corner2, size, size2=size2, overlap_d=overlap_d)
    BASE = int(v1.max() + 1)
    composite_map = v1[sel1] + v2[sel2] * BASE  # note the label width
    if composite_map.size==0:
        print("Not adjacent, not mergeable!")
        return None, None, None
    compo_idx, cnt = np.unique(composite_map, return_counts=True)
    idx2, idx1 = np.divmod(compo_idx, BASE)

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

    consensus_merge = list(set(merge_list_1) & set(merge_list_2))
    consensus_size_list = [(size_list_1[merge_list_1.index(pair)], size_list_2[merge_list_2.index(pair)]) for pair in
                           consensus_merge]
    mask = [1 if (size_pair[1] > threshold and size_pair[0] > threshold) else 0 for size_pair in consensus_size_list]
    # %% merge and remap index
    merge_array = np.array(consensus_merge)
    merge_array_filt = merge_array[np.array(mask, dtype=bool), :]
    overlap_size_array = np.array(consensus_size_list)[np.array(mask, dtype=bool)]
    overlap_size_array = overlap_size_array[:, 1]  # to 1d array
    global_shift = BASE
    v2_new = v2 + global_shift  # remap by offset
    for id1, id2 in merge_array_filt[:, :]:
        v2_new[v2_new == id2 + global_shift] = id1  # merge. (background is merged in this step)
    return merge_array_filt, overlap_size_array, v2_new
#%%
# merge_array, overlap_size_array, v2_new = merge_segment(v1, v2, (0,0,0),(0,448,0),size=(152,512,512))
#%% Generate segment list and merge_pair list!
def stitich_subvolume_grid(seg_dir, x_step, y_step, x_num, y_num, size, output_dir=None):
    x_margin = (size[2] - x_step) // 2
    y_margin = (size[1] - y_step) // 2
    seg_id_dict = []
    merge_pair_list = []
    for i in range(x_num):
        for j in range(y_num):
            corner = (0, j*y_step, i*x_step)
            f = np.load(subvolume_path(seg_dir, corner, 'npz'))
            vol = f['segmentation']
            f.close()
            idx_list = np.unique(vol)
            seg_id_dict.extend([(i, j, label) for label in idx_list])
            if i == 0:
                pass
            else:
                corner1 = (0, j*y_step, (i - 1)*x_step)
                f = np.load(subvolume_path(seg_dir, corner1, 'npz'))
                v1 = f['segmentation']
                f.close()
                merge_array, overlap_size_array, _ = merge_segment(v1, vol, corner1, corner, size, overlap_d=3, threshold=100)
                merge_pair_list.extend(
                    [[seg_id_dict.index((i - 1, j, id1)), seg_id_dict.index((i, j, id2))] for id1, id2 in merge_array])
            if j == 0:
                pass
            else:
                corner1 = (0, (j - 1)*y_step, i*x_step)
                f = np.load(subvolume_path(seg_dir, corner1, 'npz'))
                v1 = f['segmentation']
                f.close()
                merge_array, overlap_size_array, _ = merge_segment(v1, vol, corner1, corner, size, overlap_d=3, threshold=100)
                merge_pair_list.extend(
                    [[seg_id_dict.index((i, j - 1, id1)), seg_id_dict.index((i, j, id2))] for id1, id2 in merge_array])
            # full_segment[:, global_y_sel(j), global_x_sel(i)] = vol[:, local_y_sel(j), local_x_sel(i)]
    #%% find the network component in this global network!
    segment_graph = networkx.Graph()
    segment_graph.add_edges_from(merge_pair_list)
    segment_graph.add_nodes_from(range(len(seg_id_dict)))
    final_idx = []
    for component in networkx.connected_components(segment_graph):
        final_idx.append(min(component))
    #%%
    def global_x_sel(i):
        if i == 0:
            return slice(0, x_step + x_margin)
        elif i == x_num - 1:
            return slice((x_num - 1) * x_step + x_margin, x_num * x_step + 2 * x_margin)
        else:
            return slice((i - 1) * x_step + x_margin, i * x_step - x_margin)

    def global_y_sel(i):
        if i == 0:
            return slice(0, y_step + y_margin)
        elif i == y_num - 1:
            return slice((y_num - 1) * y_step + y_margin, y_num * y_step + 2 * y_margin)
        else:
            return slice((i - 1) * y_step + y_margin, i * y_step - y_margin)

    def local_x_sel(i):
        if i == 0:
            return slice(0, -x_margin)
        elif i == x_num - 1:
            return slice(x_margin, None)
        else:
            return slice(x_margin, -x_margin)

    def local_y_sel(i):
        if i == 0:
            return slice(0, -y_margin)
        elif i == y_num - 1:
            return slice(y_margin, None)
        else:
            return slice(y_margin, -y_margin)
    full_segment = np.zeros((size[0], y_num * y_step + 2 * y_margin, x_num * x_step + 2 * x_margin), dtype=np.uint32)
    for i in range(x_num):
        for j in range(y_num):
            corner = (0, j*y_step, i*x_step)
            f = np.load(subvolume_path(seg_dir, corner, 'npz'))
            vol = f['segmentation']
            f.close()
            idx_list = np.unique(vol)
            for id_loc in idx_list:
                id_glob = seg_id_dict.index((i, j, id_loc))
                equiv_group = networkx.node_connected_component(segment_graph, id_glob)
                id_glob = min(equiv_group)
                vol[vol == id_loc] = id_glob
            full_segment[:, global_y_sel(j), global_x_sel(i)] = vol[:, local_y_sel(j), local_x_sel(i)]
    if output_dir:
        seg_path = storage.segmentation_path(output_dir, (0, 0, 0) )  # FIXME: Use beg_corner instead
        storage.save_subvolume(full_segment, (0, 0, 0), seg_path)
    return full_segment, segment_graph

if __name__=="__main__":
    # Example usage
    seg_dir = "/home/morganlab/Documents/ixP11LGN/p11_1_exp10_consensus_rev/"
    full_segment, segment_graph = stitich_subvolume_grid(seg_dir, x_step=448, y_step=448, x_num=2, y_num=2,
                                                         size=(152, 512, 512),
                                                         output_dir="/home/morganlab/Documents/ixP11LGN/p11_1_exp10_full")
    seg_dict = {"seg_full": {"corner": (0, 0, 0), "vol": full_segment}}
    image_dir = "/home/morganlab/Documents/ixP11LGN/grayscale_ixP11_1_norm.h5"
    neuroglancer_visualize(seg_dict, image_dir)

    #%% Experiment 1
    seg_dir = "/home/morganlab/Documents/ixP11LGN/p11_3_exp1/"
    full_segment, segment_graph = stitich_subvolume_grid(seg_dir, x_step=448, y_step=448, x_num=2, y_num = 2, size=(152, 512, 512))
    #%%
    seg_dict = {"seg_full": {"corner": (0, 0, 0), "vol": full_segment}}
    image_dir = "/home/morganlab/Documents/ixP11LGN/grayscale_ixP11_3_norm.h5"
    neuroglancer_visualize(seg_dict, image_dir)



    #%%  Experiment 2
    seg_dir = "/home/morganlab/Documents/ixP11LGN/p11_1_exp10_consensus_rev/"
    full_segment, segment_graph = stitich_subvolume_grid(seg_dir, x_step=448, y_step=448, x_num=2, y_num=2, size=(152, 512, 512),
                                                         output_dir="/home/morganlab/Documents/ixP11LGN/p11_1_exp10_full")
    seg_dict = {"seg_full": {"corner": (0, 0, 0), "vol": full_segment}}
    image_dir = "/home/morganlab/Documents/ixP11LGN/grayscale_ixP11_1_norm.h5"
    neuroglancer_visualize(seg_dict, image_dir)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%% Garbage sites!!!!!
    BASE = int(v1.max()+1)
    composite_map = v1[:, -35,-29, :] + v2[:, 29:35, :] * BASE # note the label width
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
    mask = [1 if (size_pair[1] > threshold and size_pair[0]>threshold) else 0 for size_pair in consensus_size_list]
    #%% merge and remap index
    merge_array = np.array(consensus_merge)
    merge_array_filt = merge_array[np.array(mask, dtype=bool), :]
    overlap_size_array = np.array(consensus_size_list)[np.array(mask, dtype=bool)]
    overlap_size_array = overlap_size_array[:, 1]  # to 1d array
    #%%

    merge_array, overlap_size_array, v2_new = merge_segment(v1, v2, (0, 0, 0), (0, 448, 0), size=(152, 512, 512))

    #%%
    full_segment = np.zeros((size[0], y_num * y_step + 2 * y_margin, x_num * x_step + 2 * x_margin),dtype=np.uint16)
    for i in range(x_num):
        for j in range(y_num):
            corner = (0, j*y_step, i*x_step)
            f = np.load(subvolume_path(seg_dir, corner, 'npz'))
            vol = f['segmentation']
            f.close()
            if i==0:
                pass
            else:
                corner1 = (0, j*y_step, (i - 1)*x_step)
                f = np.load(subvolume_path(seg_dir, corner1, 'npz'))
                v1 = f['segmentation']
                f.close()
                merge_array, overlap_size_array, vol_new = merge_segment(v1, vol, corner1, corner, size, overlap_d=3, threshold=100)
                vol = vol_new
            if j==0:
                pass
            else:
                corner1 = (0, (j - 1)*y_step, i*x_step)
                f = np.load(subvolume_path(seg_dir, corner1, 'npz'))
                v1 = f['segmentation']
                f.close()
                merge_array, overlap_size_array, vol_new = merge_segment(v1, vol, corner1, corner, size, overlap_d=3, threshold=100)
                vol = vol_new
            full_segment[:, global_y_sel(j), global_x_sel(i)] = vol[:, local_y_sel(j), local_x_sel(i)]
