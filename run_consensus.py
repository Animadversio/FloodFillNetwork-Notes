import os
import time

import numpy as np
import matplotlib.pylab as plt

from absl import app
from absl import flags
from tensorflow import gfile
from google.protobuf import text_format

# from ffn.utils import bounding_box_pb2
# from ffn.inference import inference
# from ffn.inference import inference_pb2
# from ffn.inference import inference_flags
# from ffn.inference import resegmentation
from ffn.inference import storage
from ffn.inference import consensus
from ffn.inference import consensus_pb2
from ffn.inference.segmentation import make_labels_contiguous
from importlib import reload
import argparse
import ast
import logging
logging.getLogger().setLevel(logging.INFO)

def run_save_consensus(config, corners):
    consensus_req = consensus_pb2.ConsensusRequest()
    _ = text_format.Parse(config, consensus_req)
    if type(corners) is tuple:
        corner_list = [corners]
    else:
        corner_list = corners
    result = []
    for corner in corner_list:
        cons_seg, origin = consensus.compute_consensus(corner, consensus_req)
        relabel_cons_seg, relabel_map = make_labels_contiguous(cons_seg)
        # if we just relabel and discard the mapping relationship

        seg_path = storage.segmentation_path(consensus_req.segmentation_output_dir, corner)
        storage.save_subvolume(relabel_cons_seg, origin, seg_path)
        result.append(relabel_cons_seg)
    if len(corner_list)==1:
        return result[0]
    else:
        return result

ap = argparse.ArgumentParser()
ap.add_argument(
    '--config',
    help='config proto of ')
ap.add_argument(
    '--corners', help='Obtain the Neuroglancer client code from the specified URL.')
#%%
#%%
if __name__=="__main__":
    config = """
    segmentation1 {
        directory: "/home/morganlab/Documents/ixP11LGN/p11_1_exp2/"
        threshold: 0.6
        split_cc: 1
        min_size: 5000
    }
    segmentation2 {
        directory: "/home/morganlab/Documents/ixP11LGN/p11_1_exp3/"
        threshold: 0.6
        split_cc: 1
        min_size: 5000
    }
    segmentation_output_dir: "/home/morganlab/Documents/ixP11LGN/p11_1_consensus_2_3/"
    type: CONSENSUS_SPLIT
    split_min_size: 5000
    """
    corners = (0, 0, 0)
    args = ap.parse_args()
    if args.config:
        config = args.config
    if args.corners:
        corners = ast.literal_eval(args.corners)

    cons_seg = run_save_consensus(config, corners=corners)
    #%%
    # idx, cnt=np.unique(cons_seg, return_counts=True)
    #
    # #%% Self method
    # f=np.load("/home/morganlab/Documents/ixP11LGN/p11_1_exp2/0/0/seg-0_0_0.npz")
    # v1 = f['segmentation']
    # f.close()
    # f = np.load("/home/morganlab/Documents/ixP11LGN/p11_1_exp3/0/0/seg-0_0_0.npz")
    # v2 = f['segmentation']
    # f.close()
    # v1 = consensus.compute_consensus_for_segmentations(v1, v2, consensus_req)

    #%%
