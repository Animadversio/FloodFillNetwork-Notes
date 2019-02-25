"""
Utility function to visualize a bunch of segmentations overlaid on EM stack
Useful for inspecting the quality of EM stack (alignment, contrast), and the quality
of segmentation.
Can  render some neurite branches to 3d quite fast.
"""
from __future__ import print_function

import argparse
import numpy as np
from analysis_script.utils_format_convert import read_image_vol_from_h5
import neuroglancer
from ffn.inference.storage import subvolume_path
import os
import re
from  os.path import join
from ffn.inference.storage import subvolume_path
#%%
# ap = argparse.ArgumentParser()
# ap.add_argument(
#     '-a',
#     '--bind-address',
#     help='Bind address for Python web server.  Use 127.0.0.1 (the default) to restrict access '
#     'to browers running on the local machine, use 0.0.0.0 to permit access from remote browsers.')
# ap.add_argument(
#     '--static-content-url', help='Obtain the Neuroglancer client code from the specified URL.')
# args = ap.parse_args()
# if args.bind_address:
#     neuroglancer.set_server_bind_address(args.bind_address)
# if args.static_content_url:
#     neuroglancer.set_static_content_source(url=args.static_content_url)

#%%

def neuroglancer_visualize(seg_dict, image_stack, voxel_size=(8, 8, 40)):
    seg_list = []
    if "seg_dir" in seg_dict:
        seg_dir = seg_dict["seg_dir"]
        print("Visualize segmentation in %s" % seg_dir)
        for name, spec in seg_dict.items():
            if name == "seg_dir":
                continue
            corner = spec["corner"] if "corner" in spec else (0, 0, 0)
            f = np.load(subvolume_path(seg_dir, corner, 'npz'))
            vol = f['segmentation']
            f.close()
            if vol.dtype == np.uint8:
                vol = np.uint64(vol)  # add this in case the system think vol is a image
            seg_list.append( (name, corner, vol.copy()) )
    else:
        for name, spec in seg_dict.items():
            corner = spec["corner"] if "corner" in spec else (0, 0, 0)
            if "vol" in spec:
                vol = spec["vol"]
            else:
                seg_dir = spec["seg_dir"] # "/home/morganlab/Documents/ixP11LGN/p11_1_exp8"
                f = np.load(subvolume_path(seg_dir, corner, 'npz'))
                vol = f['segmentation']
                f.close()
            if vol.dtype == np.uint8: # if not change it will be identified as EM images
                vol = np.uint64(vol)
            seg_list.append((name, corner, vol)) # .copy()

    EM_stack_name = 'EM_image'
    corner = (0, 0, 0)
    if type(image_stack) is str:
        assert '.h5' in image_stack
        EM_stack = read_image_vol_from_h5(image_stack)
        seg_list.append((EM_stack_name, corner, EM_stack))
    elif type(image_stack) is np.ndarray:
        EM_stack = image_stack
        seg_list.append((EM_stack_name, corner, EM_stack))
    elif type(image_stack) is dict:
        assert len(image_stack) == 1
        for name, spec in image_stack.items():
            EM_stack_name = name
            if "vol" in spec:
                if type(spec["vol"]) is np.ndarray:
                    EM_stack = spec["vol"]
                else:
                    assert '.h5' in spec["vol"]
                    EM_stack = read_image_vol_from_h5(spec["vol"])
            if "corner" in spec:
                corner=spec["corner"]
            if "size" in spec:
                # crop the image with size
                end_corner = tuple([corner[i]+spec["size"][i] for i in range(3)])
                EM_stack = EM_stack[corner[0]:end_corner[0],
                              corner[1]:end_corner[1],
                              corner[2]:end_corner[2]]
        seg_list.append((EM_stack_name, corner, EM_stack))

    elif image_stack is None:
        pass
    else:
        raise TypeError

    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        s.voxel_size = voxel_size
        for name, corner, vol in seg_list:
            s.layers.append(
                name=name,
                layer=neuroglancer.LocalVolume(
                    data=vol,
                    # offset is in nm, not voxels, also note the difference in sequence x, y, z
                    offset=(corner[2]*voxel_size[0], corner[1]*voxel_size[1], corner[0]*voxel_size[2]),
                    voxel_size=s.voxel_size,
                ), )
        # s.layers.append(
        #     name=EM_stack_name,
        #     layer=neuroglancer.LocalVolume(
        #         data=EM_stack,
        #         voxel_size=s.voxel_size,
        #     ), )

    print(viewer)
    return viewer

def generate_seg_dict_from_dir(seg_dir, seg_name="seg"):
    '''Get all the subvolumes for neuroglancer to visualize! '''
    fn_list = sorted(os.listdir(seg_dir))
    corner_list = []
    for name in fn_list:
        if "Autoseg" in name or "counters" in name:
            continue
        cur_path = join(seg_dir, name)
        if os.path.isdir(cur_path):
            cur_x = int(name)
            for name_y in sorted(os.listdir(cur_path)):
                cur_path2 = join(cur_path, name_y)
                if os.path.isdir(cur_path2):
                    cur_y = int(name_y)
                    for fn in sorted(os.listdir(cur_path2)):
                        match = re.search("seg-%d_%d_(.*).npz" % (cur_x, cur_y), fn, )
                        if not match==None:
                            name_z = match.group(1)
                            cur_z = int(name_z)
                            if os.path.exists(subvolume_path(seg_dir, (cur_z, cur_y, cur_x), 'npz')):
                                corner_list.append((cur_z, cur_y, cur_x))
        else:
            continue
    if seg_dir[-1] == '/':
        seg_name = os.path.split(seg_dir[:-1])[1]
    else:
        seg_name = os.path.split(seg_dir)[1]
    corner_list = sorted(corner_list)
    seg_dict = {}
    seg_dict["seg_dir"] = seg_dir
    for i, corner in enumerate(corner_list):
        patch_name = seg_name +"-%d"%(i+1)
        seg_dict[patch_name] = {"corner": corner}
    return seg_dict

def merge_seg_dicts(seg_dict_list):
    output_dict = {}
    for seg_dict in seg_dict_list:
        if "seg_dir" in seg_dict:
            seg_dir = seg_dict["seg_dir"]
            for name, spec in seg_dict.items():
                if name=="seg_dir":
                    continue
                spec["seg_dir"] = seg_dir
                output_dict[name] = spec
        else:
            for name, spec in seg_dict.items():
                output_dict[name] = spec
    return output_dict

def generate_seg_dict_from_dir_list(path, seg_dir_list):
    dict_list = []
    for seg_dir_name in seg_dir_list:
        cur_dir = join(path, seg_dir_name)
        dict_list.append(generate_seg_dict_from_dir(cur_dir, seg_name=seg_dir_name))
    return merge_seg_dicts(dict_list)

#%%
if __name__=="__main__":
    # %% raw neuroglancer function example
    f = np.load("/home/morganlab/Documents/ixP11LGN/p11_1_exp4/0/0/seg-0_0_0.npz")
    v1 = f['segmentation']
    f.close()
    f = np.load("/home/morganlab/Documents/ixP11LGN/p11_1_exp5/0/0/seg-0_0_0.npz")
    v2 = f['segmentation']
    f.close()
    f = np.load("/home/morganlab/Documents/ixP11LGN/p11_1_consensus_2_3/0/0/seg-0_0_0.npz")
    cons_seg = f['segmentation']
    f.close()
    # %%
    image_stack = read_image_vol_from_h5("/home/morganlab/Documents/ixP11LGN/grayscale_ixP11_1_norm.h5")
    # %%
    f = np.load("/home/morganlab/Documents/ixP11LGN/p11_2_exp1/0/0/seg-0_0_0.npz")
    v1 = f['segmentation']
    f.close()
    f = np.load("/home/morganlab/Documents/ixP11LGN/p11_2_exp1/0/512/seg-0_512_0.npz")
    v2 = f['segmentation']
    f.close()
    # f=np.load("/home/morganlab/Documents/ixP11LGN/p11_1_consensus_2_3/0/0/seg-0_0_0.npz")
    # cons_seg = f['segmentation']
    # f.close()
    # %%
    image_stack = read_image_vol_from_h5("/home/morganlab/Documents/ixP11LGN/grayscale_ixP11_2_norm.h5")
    # %% Original function
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
            name='seg_exp1',
            layer=neuroglancer.LocalVolume(
                data=v1,
                # offset is in nm, not voxels
                # offset=(200, 300, 150),
                voxel_size=s.voxel_size,
            ), )
        s.layers.append(
            name='seg_exp1-3',
            layer=neuroglancer.LocalVolume(
                data=v2,
                # offset is in nm, not voxels
                offset=(0, 4096, 0),
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

    # %%
    import json

    seg_dict = {"seg1": {
        "corner": (0, 0, 0),
        "seg_dir": "/home/morganlab/Documents/ixP11LGN/p11_2_exp1"
    }, "seg2": {
        "corner": (0, 0, 0),
        "seg_dir": "/home/morganlab/Documents/ixP11LGN/p11_2_exp1"
    }
    }