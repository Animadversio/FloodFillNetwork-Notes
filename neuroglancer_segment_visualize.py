
from __future__ import print_function

import argparse
import numpy as np
from analysis_script.utils_format_convert import read_image_vol_from_h5
import neuroglancer
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
f=np.load("/home/morganlab/Documents/ixP11LGN/p11_1_exp4/0/0/seg-0_0_0.npz")
v1 = f['segmentation']
f.close()
f=np.load("/home/morganlab/Documents/ixP11LGN/p11_1_exp5/0/0/seg-0_0_0.npz")
v2 = f['segmentation']
f.close()
f=np.load("/home/morganlab/Documents/ixP11LGN/p11_1_consensus_2_3/0/0/seg-0_0_0.npz")
cons_seg = f['segmentation']
f.close()
#%%
image_stack = read_image_vol_from_h5("/home/morganlab/Documents/ixP11LGN/grayscale_ixP11_1_norm.h5")
#%%
f=np.load("/home/morganlab/Documents/ixP11LGN/p11_2_exp1/0/0/seg-0_0_0.npz")
v1 = f['segmentation']
f.close()
f=np.load("/home/morganlab/Documents/ixP11LGN/p11_2_exp1/0/512/seg-0_512_0.npz")
v2 = f['segmentation']
f.close()
# f=np.load("/home/morganlab/Documents/ixP11LGN/p11_1_consensus_2_3/0/0/seg-0_0_0.npz")
# cons_seg = f['segmentation']
# f.close()
#%%
image_stack = read_image_vol_from_h5("/home/morganlab/Documents/ixP11LGN/grayscale_ixP11_2_norm.h5")
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
