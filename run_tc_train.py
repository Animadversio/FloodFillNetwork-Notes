
from scipy.misc import imresize
import matplotlib.pylab as plt
from glob import glob
import numpy as np
#%%
def zero_corrected_countless(data):
    """
    Vectorized implementation of downsampling a 2D
    image by 2 on each side using the COUNTLESS algorithm.

    data is a 2D numpy array with even dimensions.
    """
    # allows us to prevent losing 1/2 a bit of information
    # at the top end by using a bigger type. Without this 255 is handled incorrectly.

    data = data + 1  # don't use +=, it will affect the original data.

    sections = []

    # This loop splits the 2D array apart into four arrays that are
    # all the result of striding by 2 and offset by (0,0), (0,1), (1,0),
    # and (1,1) representing the A, B, C, and D positions from Figure 1.
    factor = (2, 2)
    for offset in np.ndindex(factor):
        part = data[tuple(np.s_[o::f] for o, f in zip(offset, factor))]
        sections.append(part)

    a, b, c, d = sections

    ab = a * (a == b)  # PICK(A,B)
    ac = a * (a == c)  # PICK(A,C)
    bc = b * (b == c)  # PICK(B,C)

    a = ab | ac | bc  # Bitwise OR, safe b/c non-matches are zeroed

    result = a + (a == 0) * d - 1  # a or d - 1

    return result
#%% Resize Downscaling
img_dir = "/home/morganlab/Documents/ixP11LGN/TissueClassifier_Soma/Train_Img/"
seg_dir = "/home/morganlab/Documents/ixP11LGN/TissueClassifier_Soma/Label/"
# img_list = sorted(glob(img_dir+"Soma_s*.png"))
# for fn in img_list:
#     img = plt.imread(fn)
#     img_ds = imresize(img, 0.5)
#     plt.imsave(fn[:fn.find('.png')] + "_DS.png", img_ds)
#
img_list = sorted(glob(seg_dir+"IxD_W002_invert2_tissuetype_BX_soma.vsseg_export_s*.png"))
for fn in img_list:
    img = plt.imread(fn)
    img = np.uint8(img * 255)
    img_ds = imresize(img, 0.5, interp='nearest')
    plt.imsave(fn[:fn.find('.png')] + "_DS.png", img_ds)
#%%


#%%
from tissue_classify.data_prep import pixel_classify_data_proc, pixel_classify_data_generator

processor = pixel_classify_data_proc(65, 65)
processor.prepare_volume({"Soma_DS":
                {"pattern": "Soma_s*DS", "seg_pattern": "IxD_W002_invert2_tissuetype_BX_soma.vsseg_export_s*DS"}}, save=True)
#%%
processor.create_train_coordinate(2000000)

#%%
param = {"use_coord": True,
         "label_path":"/home/morganlab/Documents/ixP11LGN/TissueClassifier_Soma/Train_dataset/labels_train_ds.npy",
         "coord_path":"/home/morganlab/Documents/ixP11LGN/TissueClassifier_Soma/Train_dataset/imgs_coords_ds.npy",
         "vol_dict":{"Soma_DS": ("/home/morganlab/Documents/ixP11LGN/TissueClassifier_Soma/Train_dataset/Soma_DS_EM.h5", 'raw')},}
generator = pixel_classify_data_generator(np.arange(int(6000000*0.8)), **param)
valid_generator = pixel_classify_data_generator(np.arange(int(6000000*0.8),None), **param)
#%%
from tissue_classify.pixel_classifier2D import pixel_classifier_2d
ps2 = pixel_classifier_2d(65, 65)
ps2.train_generator(generator, valid_generator, use_multiprocessing=True,
                    workers=6)