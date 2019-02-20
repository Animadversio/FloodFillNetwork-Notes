'''
this code is as dataloader, which will load image stacks (png) into npy format for training.

'''
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
from PIL import Image
from os.path import join
from collections import defaultdict
import matplotlib.pylab as plt
import h5py
from analysis_script.utils_format_convert import read_segmentation_from_h5, read_image_vol_from_h5

def image_stack_to_vol(pattern):
    imgs = sorted(glob.glob(pattern))
    stack_n = len(imgs)
    tmp_img = plt.imread(imgs[0])
    img_shape = tmp_img.shape[:2]
    # Read in image stacks (PNG gray scale)
    image_stacks = np.zeros( (stack_n,) +img_shape, dtype=np.uint8)
    print("Image stack shape: ")
    print( (stack_n,) + img_shape)

    for i, img_name in enumerate(imgs):
        img = plt.imread(img_name)
        if len(img.shape) == 3:
            img = (img[:, :, 0] * 255).astype(dtype=np.uint8)
        elif len(img.shape) == 2:
            img = (img * 255).astype(dtype=np.uint8)
        image_stacks[i, :, :] = img

    # f = h5py.File(output_path, "w")
    # fstack = f.create_dataset("raw", (stack_n, *img_shape,), dtype='uint8')
    # fstack[:] = image_stacks
    # f.close()
    return image_stacks

class pixel_classify_data_proc(object):
    def __init__(self, x_size, y_size,
                 proj_dir="/home/morganlab/Documents/ixP11LGN/TissueClassifier_Soma/",
                 data_path="Train_Img",
                 label_path="Label",
                 test_path="Test_Img",
                 npy_path="Train_dataset",
                 img_type="png"):
        self.x_size = x_size
        self.y_size = y_size
        self.proj_dir = proj_dir
        self.data_path = join(proj_dir, data_path)
        self.label_path = join(proj_dir, label_path)
        self.img_type = img_type
        self.test_path = join(proj_dir, test_path)
        self.npy_path = join(proj_dir, npy_path)
        for path in [self.data_path, self.label_path, self.test_path, self.npy_path]:
            os.makedirs(path, exist_ok=True)
        self.vol_data_pair = []
        self.margin = ((self.x_size - 1)//2, (self.y_size - 1)//2, 0)


    def prepare_volume(self, training_data_dict, save=False):
        for vol, spec in training_data_dict.items():
            if "h5_dir" in spec:
                volume = read_image_vol_from_h5(join(self.proj_dir, spec["h5_dir"]), dataset_name='raw')
            else:
                if "pattern" in spec:
                    pattern = join(self.data_path, spec["pattern"]+"*."+self.img_type)
                    volume = image_stack_to_vol(pattern)
                else:
                    return
            if "seg_h5_dir" in spec:
                label_vol = read_image_vol_from_h5(join(self.proj_dir, spec["seg_h5_dir"]), dataset_name='raw')
            else:
                if "seg_pattern" in spec:
                    seg_pattern = join(self.label_path, spec["seg_pattern"] + "*." + self.img_type)
                    label_vol = image_stack_to_vol(seg_pattern)
                else:
                    return
            assert volume.shape == label_vol.shape
            self.vol_data_pair.append( (volume, label_vol))
        # imgs = sorted(glob.glob(pattern))
        #
        # for imgname in imgs:

    def create_train_data(self, n_samples=1000000):
        ''' Data are defaultly loaded as images in the `data_path` '''
        x_radius = (self.x_size - 1)//2
        y_radius = (self.y_size - 1)//2
        mx, my, mz = self.margin  # Note, mz is dummy for 2D case!!!
        print('-'*30)
        print('Creating training images...')
        print('-'*30)

        totals = defaultdict(int)  # partition -> voxel count (len of the indices of the same key )
        indices = defaultdict(list)
        vol_shapes = []
        for i, (volume, label_vol) in enumerate(self.vol_data_pair):
            vol_shapes.append(volume[:, my:-my, mx:-mx].shape)
            center_label_vol = label_vol[:, my:-my, mx:-mx]
            uniques, counts = np.unique(center_label_vol, return_counts=True)
            for val, cnt in zip(uniques, counts):
                if val==1:
                    continue
                totals[val] += cnt
                indices[val].extend([(i, flat_index) for flat_index in np.flatnonzero(center_label_vol == val)])
        #cut the 1024x1024 input into 9 stiles with each size of 512x512 with overlap to avoid stich issues
        #change with different image size
        print("Label, samples pair")
        label_num = len(totals)
        n_samp_per_label = n_samples//label_num + 1
        n_sample_list = defaultdict(int)
        for k, v in totals.items():
            print(' %d: %d'% (k, v))
            n_sample_list[k] = min(v, n_samp_per_label)
            if n_samp_per_label > v:
                print("Warning: not enough sample (%d) for label %d, use %d sample instead"% (n_samp_per_label, k, v))

        print("Start collecting and shuffling ")
        train_indexes = np.concatenate([np.resize(np.random.permutation(v), (n_sample_list[k], 2)) for k, v in indices.items()], axis=0)
        np.random.shuffle(indices)
        print("Finished collecting and shuffling ")

        print("Start fetching image patches.")
        imgdatas = np.ndarray((n_samples, self.y_size, self.x_size, 1), dtype=np.uint8)
        imglabels = np.ndarray((n_samples, 1), dtype=np.uint8)

        for i in range(len(train_indexes)):
            # each size of 512x512 with overlap to avoid stich issues
            vol_i, coord_idx = train_indexes[i]
            volume, label_vol = self.vol_data_pair[vol_i]
            z, y, x = np.unravel_index(coord_idx, vol_shapes[vol_i])
            imgdatas[i, :, :, 0] = volume[z,
                                   my + y - y_radius: my + y - y_radius + self.y_size,
                                   mx + x - x_radius: mx + x - x_radius + self.x_size]
            imglabels[i, 0] = label_vol[z, y, x]
            if (i+1) % 100 == 0:
                print('Done: {0}/{1} images'.format(i+1, n_samples))

        print('loading done')
        np.save(self.npy_path + '/imgs_train.npy', imgdatas)
        np.save(self.npy_path + '/labels_train.npy', imglabels)
        print('Saving to .npy files done.')

    def load_train_data(self):
        return

    def load_test_data(self):
        return

#%%

#%%
if __name__ == "__main__":
    # 512x512 is the input size of network, it could be changed to other values
    # mydata = dataProcess(512, 512)
    # processor = pixel_classify_data_proc(65, 65)
    # # mydata.create_train_data()
    # processor.prepare_volume(training_data_dict={
    #     "soma": {"pattern": "Soma_s", "seg_pattern": "IxD_W002_invert2_tissuetype_BX_soma.vsseg_export_s"}})
    # processor.create_train_data()
    processor = pixel_classify_data_proc(65, 65)
    processor.prepare_volume(
        {"soma": {"h5_dir": "Train_dataset/soma_EM.h5", "seg_h5_dir": "Train_dataset/soma_seg.h5", }})
    processor.create_train_data()