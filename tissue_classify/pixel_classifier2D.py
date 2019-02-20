import os
from os.path import join

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import keras as ks
import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)
config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 80}, gpu_options=gpu_options)
sess = tf.Session(config=config)
ks.backend.set_session(sess)

import numpy as np

from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, add, concatenate, Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from tissue_classify.data_prep import pixel_classify_data_proc
from keras.utils import np_utils


# Note merge is deprecated in Keras 2, add, concatenate is used
def merge(input, mode='concat', concat_axis=3):
    return concatenate(input, axis=concat_axis)


class pixel_classifier_2d(object):

    def __init__(self, img_rows=65, img_cols=65,
                 proj_dir="/home/morganlab/PycharmProjects/3C-LSTM-UNet/membrane_detection/"):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.proj_dir = proj_dir

    def load_traindata(self):

        mydata = pixel_classify_data_proc(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        return imgs_train, imgs_mask_train

    def load_testdata(self):

        mydata = pixel_classify_data_proc(self.img_rows, self.img_cols)
        imgs_test = mydata.load_test_data()
        return imgs_test

    def get_net(self):
        # define pixel classifier network structure with 2D input patches
        inputs = Input((self.img_rows, self.img_cols, 1))
        conv1 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(inputs)
        print("conv1 shape:", conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print("pool1 shape:", pool1.shape)

        conv2 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(pool1)
        print("conv2 shape:", conv2.shape)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print("pool2 shape:", pool2.shape)

        conv3 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(pool2)
        print("conv3 shape:", conv3.shape)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print("pool3 shape:", pool3.shape)

        conv4 = Conv2D(16, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(pool3)
        print("conv4 shape:", conv4.shape)
        # conv4 = Flatten()(conv4)
        fc = Conv2D(1, 1, activation='relu', padding='valid', kernel_initializer='he_normal')(conv4)
        print("fc shape:", fc.shape)
        fc = Flatten()(fc)
        # fc = Dense(units=512, activation='relu')(conv4)
        print("fc shape:", fc.shape)
        output = Dense(units=6, activation='softmax')(fc)
        print("output shape:", output.shape)
        model = Model(input=inputs, output=output)

        model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy',
                      metrics=['accuracy'])  # , options = run_opts)
        # loss function `binary_crossentropy` or `categorical_crossentropy` here seems binary

        return model

    # train the network with train dataset
    def train(self):
        print("loading data")
        imgs_train, labels_train = self.load_traindata()  # label_train is a vector of same length
        onehot_labels = np_utils.to_categorical(labels_train, num_classes=6)
        imgs_test = self.load_testdata()
        print("loading data done")
        model = self.get_net()
        print("got network")
        # checkponit to store the network parameter as .hdf5 file, change the name for different files saved
        pattern = "net_LGN_mb-{epoch:02d}-{val_acc:.2f}.hdf5"  # -{epoch:02d}-{val_acc:.2f}
        model_checkpoint = ModelCheckpoint(pattern, monitor='loss', verbose=1, save_best_only=True)
        # model.load_weights('unet_LGN_mb.hdf5')
        print("Weight value loaded")
        print('Fitting model...')
        model.fit(imgs_train, onehot_labels, batch_size=16, epochs=20, verbose=1, validation_split=0.2, shuffle=True,
                  callbacks=[model_checkpoint])

    # test the network with test dataset
    def load_trained_model(self, checkpoint_path=None):
        print("loading data")
        imgs_test = self.load_testdata()
        print("loading data done")
        model = self.get_net()
        print("got network")
        # load checkpoint to store the network parameter as .hdf5 file, change the name for different files saved
        if checkpoint_path==None:
            checkpoint_path = 'unet_LGN_mb.hdf5'
        model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True)
        model.load_weights(checkpoint_path)
        print('load trained model')
        print('predict test data')
        imgs_mask_test = model.predict(imgs_test, batch_size=5, verbose=1)
        # create new folder to store the results
        np.save(join(self.proj_dir, "LGN_train/trainborders/imgs_mask_test.npy"), imgs_mask_test)

    # def inference(self):
    #
    # def save_img(self):
    #     # should change with respect to image size
    #     print("array to image")
    #     imgs = np.load(join(self.proj_dir, "LGN_train/trainborders/imgs_mask_test.npy"))
    #     orig_shape = (1058, 1180)
    #     r_step = self.img_rows // 2
    #     c_step = self.img_cols // 2
    #     x_num = 1 + (orig_shape[0] - self.img_rows) // r_step
    #     y_num = 1 + (orig_shape[1] - self.img_cols) // c_step
    #     # coordinate on the big canvas
    #     x1 = np.arange(x_num) * r_step + r_step // 2
    #     x2 = np.arange(x_num) * r_step + self.img_rows - r_step // 2  # Specify the lower and upper bound of image piece
    #     y1 = np.arange(y_num) * c_step + c_step // 2
    #     y2 = np.arange(y_num) * c_step + self.img_cols - c_step // 2  # Specify the lower and upper bound of image piece
    #     x1[0] = 0
    #     y1[0] = 0
    #     x2[-1] = (x_num - 1) * r_step + self.img_rows
    #     y2[-1] = (y_num - 1) * c_step + self.img_cols
    #     Imag = np.ndarray((x2[-1], y2[-1], 1), dtype=np.uint8)
    #     # coordinate pairs on loaded image
    #     lx1 = np.ones(x_num, dtype=np.int32) * self.img_rows // 4
    #     lx2 = np.ones(x_num, dtype=np.int32) * self.img_rows // 4 * 3
    #     ly1 = np.ones(y_num, dtype=np.int32) * self.img_cols // 4
    #     ly2 = np.ones(y_num, dtype=np.int32) * self.img_cols // 4 * 3
    #     lx1[0] = 0
    #     ly1[0] = 0
    #     lx2[-1] = self.img_rows
    #     ly2[-1] = self.img_cols
    #     assert r_step == c_step  # FIXME: here can be generalized to different r, c
    #     # stich the tiles
    #     for i in range(imgs.shape[0]):  # assume the images are not
    #         # id of image patch
    #         ix = (i % (x_num * y_num)) // y_num
    #         iy = (i % (x_num * y_num)) % y_num
    #         img = imgs[i]
    #         Imag[x1[ix]:x2[ix], y1[iy]:y2[iy], 0] = img[lx1[ix]:lx2[ix], ly1[iy]:ly2[iy], 0] * 255
    #         if (i + 1) % (x_num * y_num) == 0:
    #             result = array_to_img(Imag)
    #             result.save(join(self.proj_dir, "LGN_train/trainborders/test_%04d.png" % (i // (x_num * y_num))))
    #             Imag = np.ndarray((x2[-1], y2[-1], 1), dtype=np.uint8)


if __name__ == '__main__':
    pc2 = pixel_classifier_2d(img_rows=128, img_cols=128)
    # network train
    pc2.train()
    # network inference
    pc2.load_trained_model()
    pc2.save_img()
