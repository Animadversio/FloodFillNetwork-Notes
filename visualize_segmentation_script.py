from absl import flags
from absl import app
from ffn.utils import bounding_box_pb2
from ffn.inference import inference_flags
from google.protobuf import text_format
from tensorflow import gfile
from os.path import join

import matplotlib as mpl
import numpy as np
mpl.use('Agg')
from analysis_script.read_segmentation_results import load_segmentation_output, visualize_supervoxel_size_dist, \
    export_segmentation_to_VAST, export_composite_image
from analysis_script.utils_format_convert import read_image_vol_from_h5
FLAGS = flags.FLAGS

# Options related to training data.
# flags.DEFINE_string('seg_dir', None,
#                     'Glob for the TFRecord of training coordinates.') # What's the use of TFRecord
# flags.DEFINE_string('seg_export_dir', None, '')
# flags.DEFINE_string('render_dir', None, '')
# flags.DEFINE_string('imageh5_dir', None, '')
# exportLoc = '/home/morganlab/Documents/Autoseg_result/LGN_Autoseg_full2'
flags.DEFINE_string('bounding_box', None,
                    'BoundingBox proto in text format defining the area '
                    'to segmented.')
flags.DEFINE_bool('visualize', False, 'Output rendered files (segmentation composite with image)')
flags.DEFINE_float('resize', 1.0, '')
flags.DEFINE_bool('stat_only', True, 'No Output rendered files (segmentation composite with image)')
FLAGS = flags.FLAGS

def main(unused_argv):
    request = inference_flags.request_from_flags()
    bbox = bounding_box_pb2.BoundingBox()  # bounding box structure
    text_format.Parse(FLAGS.bounding_box, bbox)
    seg_dir = request.segmentation_output_dir
    # FLAGS.seg_dir  # '/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR_Longtime_success2/'  # Longterm wide field, lowthreshold file
    corner = (bbox.start.z, bbox.start.y, bbox.start.x)
    up_corner = (bbox.start.z + bbox.size.z, bbox.start.y + bbox.size.y, bbox.start.x + bbox.size.x,)
    segmentation, qprob = load_segmentation_output(seg_dir, corner)
    print("Segmentation loaded!")
    # %%
    export_dir = join(seg_dir, "Autoseg-%s" % ('_'.join([str(x) for x in corner[::-1]])))
    if not gfile.Exists(export_dir):
        gfile.MakeDirs(export_dir)
    #%%
    idx, cnts = visualize_supervoxel_size_dist(segmentation, save_dir=export_dir, save_fig=True, show_fig=False)
    print("Supervoxel statistics visualization complete!")
    print("Labels count:%d, Mean size:%d, Median size:%d" % (len(idx), np.nanmean(cnts), np.nanmedian(cnts)))
    if FLAGS.stat_only:
        return
    else:
        # %%
        export_segmentation_to_VAST(export_dir, segmentation, resize=FLAGS.resize)
        # canvas.segmentation  segmentation  np.nan_to_num(canvas.seed>0.6)
        print("Export to VAST complete!")
        #%%
        if FLAGS.visualize:
            imageh5_dir, dataset_name = request.image.hdf5.split(":")
            image_stack = read_image_vol_from_h5(imageh5_dir)
            export_composite_image(segmentation, image_stack, export_dir, suffix="png",
                                   alpha=0.2, resize=1, bbox=[corner, up_corner])
            print("Export rendered image complete!")
#%%
if __name__=="__main__":
    app.run(main)

