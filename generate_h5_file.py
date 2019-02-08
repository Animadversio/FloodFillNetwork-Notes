from analysis_script.utils_format_convert import convert_image_stack_to_h5, normalize_img_stack
from absl import app
from absl import flags
import numpy as np
FLAGS = flags.FLAGS
flags.DEFINE_string('path', None, 'path of image files')
flags.DEFINE_string('name_pattern', None, "Name pattern of image files")
flags.DEFINE_string('output_name', "grayscale_maps_LR.h5", "Name of h5 files")
flags.DEFINE_integer('stack_n', 175, "number that seed coordinate have to divide to be the subvolume coordinate")
flags.DEFINE_integer('beg_n', 0, "starting id")
FLAGS = flags.FLAGS


def main(unused):
    beg_n = FLAGS.beg_n
    path = FLAGS.path
    EM_name_pattern = FLAGS.name_pattern# "tweakedImageVolume2_LRexport_s%03d.png"
    stack_n = FLAGS.stack_n
    # raw_name_pattern = "Segmentation1-LX_8-14.vsseg_LRexport_s%03d_1184x1072_16bpp.raw"
    EM_stack = convert_image_stack_to_h5(path=path, pattern=EM_name_pattern, stack_n=stack_n, beg_n=beg_n,
                                     output=FLAGS.output_name)
    print("mean: %.2f, std: %.2f"% (EM_stack.mean(), EM_stack.std()))
    normalize_img_stack(path, "grayscale_ixP11_3_norm.h5", EM_stack)

if __name__=="__main__":
    app.run(main)