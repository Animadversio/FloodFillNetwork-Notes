from PIL import Image
import numpy as np
import h5py
from os.path import join
from skimage.measure import block_reduce
#%% Downsampling image

def down_sample_img_stack(path, output, EM_stack, filter_func=np.median, scale=(2, 2)):
    """Use block reduce to apply median filter to image stack for each layer
    :param scale is the block shape, must be integer tuple. (can be adjusted to get 3d block reduce)"""
    if type(EM_stack) is str:
        vol_dir = join(path, EM_stack)
        f = h5py.File(vol_dir, 'r')
        image_stack = f['raw']
        EM_stack = image_stack[:]
        f.close()
    ds_image_stack = []
    for img in EM_stack:
        ds_img = block_reduce(img, scale, filter_func)
        ds_image_stack.append(ds_img)
        # Image.fromarray(ds_img).show()
    ds_image_stack = np.array(ds_image_stack)
    print("Shape after downsampling %s" % str(ds_image_stack.shape))
    f = h5py.File(join(path, output), "w")
    fstack=f.create_dataset("raw", ds_image_stack.shape, dtype='uint8') # Note they only take int64 input
    fstack[:] = ds_image_stack
    f.close()
    return ds_image_stack


def normalize_img_stack_with_mask(path, output, EM_stack, upper=196, lower=80, up_outlier=245, low_outlier=30):
    ''' Discard outlier when doing percentile matching.
    (In case a large area of black/white irregularity will affect the percentile and change the
     overall intensity/ contrast)
     '''
    low_p = []
    high_p = []
    for img in EM_stack:
        img1d = img.flatten()
        img1d = img1d[np.logical_and(img1d < up_outlier, img1d > low_outlier)]
        low_p.append(np.percentile(img1d, 5))
        high_p.append(np.percentile(img1d, 95))
    low_p = np.array(low_p)
    high_p = np.array(high_p)
    # low_p = np.percentile(EM_stack, 5, axis=[1,2])
    # high_p = np.percentile(EM_stack, 95, axis=[1,2])
    scaler = (upper-lower) / (high_p - low_p)
    shift = lower - (low_p*scaler)
    norm_img = scaler.reshape((-1, 1, 1)) * EM_stack + shift.reshape((-1, 1, 1))
    print("max: %.2f, min: %.2f after scaling"%(norm_img.max(), norm_img.min()))
    int_img = np.clip(norm_img, 0, 255, )
    int_img = int_img.astype('uint8')
    img_shape = EM_stack.shape
    f = h5py.File(join(path, output), "w")
    fstack=f.create_dataset("raw", img_shape, dtype='uint8') # Note they only take int64 input
    fstack[:] = int_img
    f.close()
    return int_img
