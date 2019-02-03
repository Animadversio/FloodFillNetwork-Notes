from absl import flags
from absl import app
from analysis_script.read_segmentation_results import load_segmentation_output, visualize_supervoxel_size_dist, \
    export_segmentation_to_VAST, export_composite_image
from analysis_script.utils_format_convert import read_image_vol_from_h5
FLAGS = flags.FLAGS

# Options related to training data.
flags.DEFINE_string('seg_dir', None,
                    'Glob for the TFRecord of training coordinates.') # What's the use of TFRecord
flags.DEFINE_string('seg_export_dir', None, '')
flags.DEFINE_string('render_dir', None, '')
flags.DEFINE_string('imageh5_dir', None, '')
flags.DEFINE_bool('visualize', False, '')
flags.DEFINE_float('resize', 1.0, '')
flags.DEFINE_string('bounding_box', None,
                    'BoundingBox proto in text format defining the area '
                    'to segmented.')
FLAGS = flags.FLAGS

def main():
    seg_dir = FLAGS.seg_dir  # '/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR_Longtime_success2/'  # Longterm wide field, lowthreshold file
    corner = (0, 0, 0)
    segmentation, qprob = load_segmentation_output(seg_dir, corner)
    # %%
    idx, cnts = visualize_supervoxel_size_dist(segmentation, save_dir=FLAGS.seg_output_dir, save_fig=True)
    # %%
    # exportLoc = '/home/morganlab/Documents/Autoseg_result/LGN_Autoseg_full2'
    export_segmentation_to_VAST(FLAGS.seg_export_dir, segmentation, resize=FLAGS.resize)
    # canvas.segmentation  segmentation  np.nan_to_num(canvas.seed>0.6)
    #%%
    if FLAGS.visualize:
        image_stack = read_image_vol_from_h5(FLAGS.imageh5_dir)
        export_composite_image(segmentation, image_stack, FLAGS.render_dir, suffix="png",
                               alpha=0.2, resize=1, bbox=[(0,0,0), (175,1058,1180)])
#%%
if __name__=="__main__":
    app.run(main)

