'''
this code is as dataloader, which will load image stacks (png) into npy format for training.

'''
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
from PIL import Image
from os.path import join, exists
from collections import defaultdict
import matplotlib.pylab as plt
import h5py
from analysis_script.utils_format_convert import read_segmentation_from_h5, read_image_vol_from_h5
import keras
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
#%%
class pixel_classify_data_proc(object):
    """Memory consuming hard storage version"""
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


    def prepare_volume(self, training_data_dict, save=True):
        '''Make images into ndarray and save as h5 file'''
        for vol, spec in training_data_dict.items():
            if "h5_dir" in spec:
                volume = read_image_vol_from_h5(join(self.proj_dir, spec["h5_dir"]), dataset_name='raw')
            else:
                if "pattern" in spec:
                    pattern = join(self.data_path, spec["pattern"]+"*."+self.img_type)
                    volume = image_stack_to_vol(pattern)
                    if save:
                        f = h5py.File(join(self.npy_path, vol+"_EM.h5"), "w")
                        fstack = f.create_dataset("raw", volume.shape, dtype='uint8')  # Note they only take int64 input
                        fstack[:] = volume
                        f.close()
                else:
                    return
            if "seg_h5_dir" in spec:
                label_vol = read_image_vol_from_h5(join(self.proj_dir, spec["seg_h5_dir"]), dataset_name='raw')
            else:
                if "seg_pattern" in spec:
                    seg_pattern = join(self.label_path, spec["seg_pattern"] + "*." + self.img_type)
                    label_vol = image_stack_to_vol(seg_pattern)
                    if save:
                        f = h5py.File(join(self.npy_path, vol+"_seg.h5"), "w")
                        fstack = f.create_dataset("raw", volume.shape, dtype='uint8')  # Note they only take int64 input
                        fstack[:] = volume
                        f.close()
                else:
                    return
            assert volume.shape == label_vol.shape
            self.vol_data_pair.append((volume, label_vol))

    def create_train_data(self, n_samples=16000000):
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
        # Quite slow
        print("Start collecting and shuffling ")
        train_indexes = np.concatenate([np.resize(np.random.permutation(v), (n_sample_list[k], 2)) for k, v in indices.items()], axis=0)
        np.random.shuffle(indices)  # slow!!!
        print("Finished collecting and shuffling ")

        print("Start fetching image patches.")
        imgdatas = np.ndarray((len(train_indexes), self.y_size, self.x_size, 1), dtype=np.uint8)
        imglabels = np.ndarray((len(train_indexes), 1), dtype=np.uint8)
        # Super-fast below!
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

    def create_train_data_sample(self, n_samples=6000000):
        ''' Data are defaultly loaded as images in the `data_path`
        (More memory efficient way )'''
        x_radius = (self.x_size - 1)//2
        y_radius = (self.y_size - 1)//2
        mx, my, mz = self.margin  # Note, mz is dummy for 2D case!!!
        print('-'*30)
        print('Creating training images...')
        print('-'*30)

        totals = defaultdict(int)  # partition -> voxel count (len of the indices of the same key )
        vol_shapes = []
        vol_size_lmt = []
        size_tmp = 0
        for i, (volume, label_vol) in enumerate(self.vol_data_pair):
            center_label_vol = label_vol[:, my:-my, mx:-mx]
            vol_shapes.append(center_label_vol.shape)
            vol_size_lmt.append(size_tmp)
            size_tmp += np.prod(center_label_vol.shape)

            uniques, counts = np.unique(center_label_vol, return_counts=True)
            for val, cnt in zip(uniques, counts):
                if val==1:  # temporarily ignore that
                    continue
                totals[val] += cnt
        tot_voxel_num = size_tmp
        print("Total voxel number to sample: %d" % tot_voxel_num)
        #cut the 1024x1024 input into 9 stiles with each size of 512x512 with overlap to avoid stich issues
        #change with different image size
        print("Label, samples pair")
        label_num = len(totals)
        n_samp_per_label = n_samples
        n_sample_list = defaultdict(int)
        sample_counter = defaultdict(int)
        for k, v in totals.items():
            print(' %d: %d' % (k, v))
            n_sample_list[k] = min(v, n_samp_per_label)
            sample_counter[k] = 0
            if n_samp_per_label > v:
                print("Warning: not enough sample (%d) for label %d, use %d sample instead" % (n_samp_per_label, k, v))
        # Quite slow
        total_samples = sum(n_sample_list.values())
        rand_points = np.random.randint(tot_voxel_num, size=total_samples)
        print("Start fetching image patches.")
        imgdatas = np.ndarray((total_samples, self.y_size, self.x_size, 1), dtype=np.uint8)
        imglabels = np.ndarray((total_samples, 1), dtype=np.uint8)
        # Super-fast below!
        for i in range(total_samples):
            # each size of 512x512 with overlap to avoid stich issues
            glob_idx = rand_points[i]
            while True:
                vol_i = min([i for i in range(len(vol_size_lmt)) if glob_idx >  vol_size_lmt[i]])
                coord_idx = glob_idx - vol_size_lmt[vol_i]
                volume, label_vol = self.vol_data_pair[vol_i]
                z, y, x = np.unravel_index(coord_idx, vol_shapes[vol_i])
                l = label_vol[z, y, x]

                if sample_counter[l] < n_sample_list[l]:
                    break
                else:
                    glob_idx = np.random.randint(tot_voxel_num)
            imgdatas[i, :, :, 0] = volume[z,
                                   my + y - y_radius: my + y - y_radius + self.y_size,
                                   mx + x - x_radius: mx + x - x_radius + self.x_size]
            imglabels[i, 0] = l
            sample_counter[l] += 1
            if (i+1) % 100 == 0:
                print('Done: {0}/{1} images'.format(i+1, total_samples))

        print('loading done')
        # np.save(self.npy_path + '/imgs_train.npy', imgdatas)
        f = h5py.File(join(self.npy_path, "imgs_train.h5"), "w")
        fstack = f.create_dataset("patch", imgdatas.shape, dtype='uint8')
        fstack[:] = imgdatas
        f.close()
        np.save(self.npy_path + '/labels_train.npy', imglabels)
        print('Saving to .npy files done.')

    def create_train_coordinate(self, n_samp_per_label=8000000):
        ''' Data are saved as ndarray of coordinates and volume image
        (More memory efficient way )'''
        # x_radius = (self.x_size - 1)//2
        # y_radius = (self.y_size - 1)//2
        mx, my, mz = self.margin  # Note, mz is dummy for 2D case!!!
        print('-'*30)
        print('Creating training coordinates...')
        print('-'*30)

        totals = defaultdict(int)  # partition -> voxel count (len of the indices of the same key )
        vol_shapes = []
        vol_size_lmt = []
        size_tmp = 0
        for i, (volume, label_vol) in enumerate(self.vol_data_pair):
            center_label_vol = label_vol[:, my:-my, mx:-mx]
            vol_shapes.append(center_label_vol.shape)
            vol_size_lmt.append(size_tmp)
            size_tmp += np.prod(center_label_vol.shape)

            uniques, counts = np.unique(center_label_vol, return_counts=True)
            for val, cnt in zip(uniques, counts):
                if val==1:  # temporarily ignore that
                    continue
                totals[val] += cnt
        tot_voxel_num = size_tmp
        print("Total voxel number to sample: %d" % tot_voxel_num)
        #cut the 1024x1024 input into 9 stiles with each size of 512x512 with overlap to avoid stich issues
        #change with different image size
        print("Label, samples pair")
        label_num = len(totals)

        n_sample_list = defaultdict(int)
        sample_counter = defaultdict(int)
        for k, v in totals.items():
            print(' %d: %d' % (k, v))
            n_sample_list[k] = min(v, n_samp_per_label)
            sample_counter[k] = 0
            if n_samp_per_label > v:
                print("Warning: not enough sample (%d) for label %d, use %d sample instead" % (n_samp_per_label, k, v))

        total_samples = sum(n_sample_list.values())
        rand_points = np.random.randint(tot_voxel_num, size=total_samples)
        print("Start fetching image patches.")
        imgcoords = np.ndarray((total_samples, 4), dtype=int)
        imglabels = np.ndarray((total_samples, ), dtype=np.uint8)
        # Super-fast below!
        for i in range(total_samples):
            # each size of 512x512 with overlap to avoid stich issues
            glob_idx = rand_points[i]
            while True:
                vol_i = min([i for i in range(len(vol_size_lmt)) if glob_idx >= vol_size_lmt[i]])
                coord_idx = glob_idx - vol_size_lmt[vol_i]
                volume, label_vol = self.vol_data_pair[vol_i]
                z, y, x = np.unravel_index(coord_idx, vol_shapes[vol_i])
                l = label_vol[z, my + y, mx + x]
                if sample_counter[l] < n_sample_list[l]:
                    break
                else:
                    glob_idx = np.random.randint(tot_voxel_num)

            imgcoords[i, :] = [vol_i, z, my + y, mx + x]
            imglabels[i] = l
            sample_counter[l] += 1
            if (i+1) % 1000 == 0:
                print('Done: {0}/{1} coordinates'.format(i+1, total_samples))

        indexes = np.arange(total_samples)
        np.random.shuffle(indexes)
        imgcoords = imgcoords[indexes, :]
        imglabels = imglabels[indexes]

        print('loading done')
        np.save(self.npy_path + '/imgs_coords.npy', imgcoords)
        np.save(self.npy_path + '/labels_train.npy', imglabels)
        print('Saving to .npy files done.')

    def load_train_data(self):
        if exists(self.npy_path + '/imgs_train.npy'):
            imgdatas = np.load(self.npy_path + '/imgs_train.npy')
        elif exists(join(self.npy_path, "imgs_train.h5")):
            f = h5py.File(join(self.npy_path, "imgs_train.h5"), "r")
            imgdatas = f["patch"]
            # f.close()
        else:
            raise FileNotFoundError
        imglabels = np.load(self.npy_path + '/labels_train.npy')
        return imgdatas, imglabels

    def load_test_data(self):
        return
#%%
class pixel_classify_data_generator(keras.utils.Sequence):
    '''Generates data for Keras
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    '''
    def __init__(self, list_IDs, batch_size=16, dim=(65, 65), n_channels=1, n_classes=6, shuffle=True, use_coord=True,
                 label_path="/home/morganlab/Documents/ixP11LGN/TissueClassifier_Soma/Train_dataset/labels_train.npy",
                 coord_path="/home/morganlab/Documents/ixP11LGN/TissueClassifier_Soma/Train_dataset/imgs_coords.npy",
                 vol_dict={"Soma": ("soma_EM.h5", 'raw')},
                 img_path="/home/morganlab/Documents/ixP11LGN/TissueClassifier_Soma/Train_dataset/imgs_train.h5"):
        'Initialization'
        self.dim = dim
        self.sy, self.sx = dim
        self.ry = (self.sy - 1) // 2
        self.rx = (self.sx - 1) // 2
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.list_IDs = list_IDs

        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

        self.use_coord = use_coord
        self.labels = np.load(label_path)
        if use_coord:
            self.coords = np.load(coord_path)
            self.vol_handle_list = []
            self.get_volome(vol_dict)
        else:
            self.f = h5py.File(img_path, "r")

    def get_volome(self, vol_dict):
        for name, (path, ds_name) in vol_dict.items():
            self.vol_handle_list.append((h5py.File(path, "r"), ds_name) )
        return

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size), dtype=np.uint8)

        # Generate data
        if not self.use_coord:
            for i, ID in enumerate(list_IDs_temp):
                # Store sample
                X[i, ] = self.f["patch"][ID, :, :, :]
                # Store class
                Y[i] = self.labels[ID]
        else:
            for i, ID in enumerate(list_IDs_temp):
                vol_i, z, y, x = self.coords[ID]
                volume, ds_name = self.vol_handle_list[vol_i]
                # Store sample
                X[i, :, :, 0] = volume[ds_name][z,
                                       y - self.ry: y - self.ry + self.sy,
                                       x - self.rx: x - self.rx + self.sx]
                # Store class
                Y[i] = self.labels[ID]
        return X, keras.utils.to_categorical(Y, num_classes=self.n_classes)


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)



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
    processor.prepare_volume(training_data_dict={
       "soma": {"pattern": "Soma_s", "seg_pattern": "IxD_W002_invert2_tissuetype_BX_soma.vsseg_export_s"}})

    # processor.prepare_volume(
    #     {"soma": {"h5_dir": "Train_dataset/soma_EM.h5", "seg_h5_dir": "Train_dataset/soma_seg.h5", }})
    processor.create_train_coordinate()

    # generator = pixel_classify_data_generator(np.arange(1000),
    #     vol_dict={"Soma":("/home/morganlab/Documents/ixP11LGN/TissueClassifier_Soma/Train_dataset/soma_EM.h5", 'raw')})
    # generator.__getitem__(1)