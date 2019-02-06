import os
import time

from google.protobuf import text_format
from absl import app
from absl import flags
import ast
from ffn.utils import bounding_box_pb2
from ffn.inference import inference
from ffn.inference import inference_pb2
from ffn.inference import storage
from scipy.special import expit, logit
from tensorflow import gfile
import numpy as np
import sys
import logging
import logging.config
#%%
logger = logging.getLogger(__name__)

FLAGS = flags.FLAGS
flags.DEFINE_string('inference_request', None,
                    'InferenceRequest proto in text format.')
flags.DEFINE_string('bounding_box', None,
                    'BoundingBox proto in text format defining the area '
                    'to segmented. In the subvolume coordinate')
flags.DEFINE_string('logfile_name', "inference_log1.log", "Log file to write the log instantaneously")
flags.DEFINE_float('downsample_factor', 1, "number that seed coordinate have to divide to be the subvolume coordinate"
                                           ", How large is the pixel in subvolume correspond to the coordinate in which "
                                           "the seed coordinate is generated")
flags.DEFINE_string('corner', "(0,0,0)", "corner of the subvolume in the full volume. "
                            "the coordinate of origin of the subvolume in full volume")
flags.DEFINE_string('seed_list', None, "a list of x,y,z tuples to start segmentation at")
FLAGS = flags.FLAGS
# seed_list = [(1080, 860, 72), (1616, 1872, 43), (612, 1528, 92), (616, 180, 92),  (144, 712, 43), (400, 168, 45), (1332, 248, 45), (120, 700,45)]  # in xyz order
# downsample_factor = 2
# canvas_bbox = [(0, 0, 0), (175, 1058, 1180)]
# corner = (0, 0, 0)

def main(unused_argv):
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(filename)s: %(funcName)s(): %(lineno)d] %(message)s"
            },
        },
        "handlers": {
            "default": {
                "level": "INFO",
                "class": "logging.StreamHandler",
            },
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
                "stream": "ext://sys.stdout"
            },
            "logfile": {
                "class": "logging.FileHandler",
                "level": "INFO",
                "formatter": "standard",
                "filename": FLAGS.logfile_name,
                "encoding": "utf8"
            }
        },
        "loggers": {
            "": {
                "handlers": ["console", "logfile"],
                "level": "INFO",
                "propagate": True
            }
        }
    })
    logging.info("Logger prepared! ")

    request = inference_pb2.InferenceRequest()
    _ = text_format.Parse(FLAGS.inference_request, request)
    if not gfile.Exists(request.segmentation_output_dir):
        gfile.MakeDirs(request.segmentation_output_dir)
    corner = ast.literal_eval(FLAGS.corner)
    downsample_factor = FLAGS.downsample_factor
    bbox = bounding_box_pb2.BoundingBox()  # bounding box structure
    text_format.Parse(FLAGS.bounding_box, bbox)
    seed_list = ast.literal_eval(FLAGS.seed_list)
    runner = inference.Runner()
    runner.start(request)
    canvas, alignment = runner.make_canvas((bbox.start.z, bbox.start.y, bbox.start.x),
                                            (bbox.size.z, bbox.size.y, bbox.size.x))
    # like (0, 0, 0), (175, 1058, 1180)
    for id, start_point in enumerate(seed_list):
        label = id + 1
        pos = (int(start_point[2]-corner[2]), int((start_point[1]-corner[1])//downsample_factor), int((start_point[0]-corner[0])//downsample_factor))
        canvas.log_info('Starting segmentation at %r (zyx)', pos)
        num_iters = canvas.segment_at(pos,)  # zyx
                        # dynamic_image=inference.DynamicImage(),
                        # vis_update_every=2, vis_fixed_z=True)
        mask = canvas.seed >= request.inference_options.segment_threshold

        raw_segmented_voxels = np.sum(mask)  # count the number of raw segmented voxels
        # Record existing segment IDs overlapped by the newly added object.
        overlapped_ids, counts = np.unique(canvas.segmentation[mask],
                                           return_counts=True)
        valid = overlapped_ids > 0  # id < 0 are the background and invalid voxels
        overlapped_ids = overlapped_ids[valid]
        counts = counts[valid]
        if len(counts) > 0:
            print('Overlapping segments (label, size): ', *zip(overlapped_ids, counts), sep=" ")

        mask &= canvas.segmentation <= 0  # get rid of the marked voxels from mask
        actual_segmented_voxels = np.sum(mask)  # only count those unmarked voxel in `actual_segmented_voxels`
        canvas.segmentation[mask] = label
        canvas.log_info('Created supervoxel:%d  seed(zyx):%s  size:%d (raw size %d)  iters:%d',
                    label, pos, actual_segmented_voxels, raw_segmented_voxels, num_iters)

        np.savez(os.path.join(request.segmentation_output_dir, 'seg%03d_%s.npz' % (label, str(start_point))),
               segmentation=mask,
                prob=storage.quantize_probability(
                expit(canvas.seed)))

    counter_path = os.path.join(request.segmentation_output_dir, 'counters.txt')
    if not gfile.Exists(counter_path):
        runner.counters.dump(counter_path)
    #%%
    # corner = (0,0,0)
    seg_path = storage.segmentation_path(
            request.segmentation_output_dir, corner)
    prob_path = storage.object_prob_path(
            request.segmentation_output_dir, corner)
    runner.save_segmentation(canvas, alignment, seg_path, prob_path)

if __name__=="__main__":
    app.run(main)