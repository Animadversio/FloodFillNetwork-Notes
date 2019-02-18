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
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, add, concatenate
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from data_k_2d import *


# Note merge is deprecated in Keras 2, add, concatenate is used
def merge(input, mode='concat', concat_axis=3):
    return concatenate(input, axis=concat_axis)


class myUnet(object):

    def __init__(self, img_rows=512, img_cols=512,
                 proj_dir="/home/morganlab/PycharmProjects/3C-LSTM-UNet/membrane_detection/"):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.proj_dir = proj_dir

    def load_traindata(self):

        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        return imgs_train, imgs_mask_train

    def load_testdata(self):

        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_test = mydata.load_test_data()
        return imgs_test

    def get_unet(self):
        # define Unet network structure with 2D input patches
        inputs = Input((self.img_rows, self.img_cols, 1))
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        print("conv1 shape:", conv1.shape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        print("conv1 shape:", conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        # print "pool1 shape:",pool1.shape

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        # print "conv2 shape:",conv2.shape
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        # print "conv2 shape:",conv2.shape
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        # print "pool2 shape:",pool2.shape

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        # print "conv3 shape:",conv3.shape
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        # print "conv3 shape:",conv3.shape
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        # print "pool3 shape:",pool3.shape

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        model = Model(input=inputs, output=conv10)

        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy',
                      metrics=['accuracy'])  # , options = run_opts)
        # loss function `binary_crossentropy` or `categorical_crossentropy` here seems binary

        return model

    # train the network with train dataset
    def train(self):
        print("loading data")
        imgs_train, imgs_mask_train = self.load_traindata()
        imgs_test = self.load_testdata()
        print("loading data done")
        model = self.get_unet()
        print("got unet")
        # checkponit to store the network parameter as .hdf5 file, change the name for different files saved
        pattern = "unet_LGN_mb-{epoch:02d}-{val_acc:.2f}.hdf5"  # -{epoch:02d}-{val_acc:.2f}
        model_checkpoint = ModelCheckpoint(pattern, monitor='loss', verbose=1, save_best_only=True)
        model.load_weights('unet_LGN_mb.hdf5')
        print("Weight value loaded")
        print('Fitting model...')
        model.fit(imgs_train, imgs_mask_train, batch_size=4, epochs=20, verbose=1, validation_split=0.2, shuffle=True,
                  callbacks=[model_checkpoint])

    # test the network with test dataset
    def load_trained_model(self):
        print("loading data")
        imgs_test = self.load_testdata()
        print("loading data done")
        model = self.get_unet()
        print("got unet")
        # load checkpoint to store the network parameter as .hdf5 file, change the name for different files saved
        model_checkpoint = ModelCheckpoint('unet_LGN_mb.hdf5', monitor='loss', verbose=1, save_best_only=True)
        model.load_weights('unet_LGN_mb.hdf5')
        print('load trained model')
        print('predict test data')
        imgs_mask_test = model.predict(imgs_test, batch_size=5, verbose=1)
        # create new folder to store the results
        np.save(join(self.proj_dir, "LGN_train/trainborders/imgs_mask_test.npy"), imgs_mask_test)

    def save_img(self):
        # should change with respect to image size
        print("array to image")
        imgs = np.load(join(self.proj_dir, "LGN_train/trainborders/imgs_mask_test.npy"))
        orig_shape = (1058, 1180)
        r_step = self.img_rows // 2
        c_step = self.img_cols // 2
        x_num = 1 + (orig_shape[0] - self.img_rows) // r_step
        y_num = 1 + (orig_shape[1] - self.img_cols) // c_step
        # coordinate on the big canvas
        x1 = np.arange(x_num) * r_step + r_step // 2
        x2 = np.arange(x_num) * r_step + self.img_rows - r_step // 2  # Specify the lower and upper bound of image piece
        y1 = np.arange(y_num) * c_step + c_step // 2
        y2 = np.arange(y_num) * c_step + self.img_cols - c_step // 2  # Specify the lower and upper bound of image piece
        x1[0] = 0
        y1[0] = 0
        x2[-1] = (x_num - 1) * r_step + self.img_rows
        y2[-1] = (y_num - 1) * c_step + self.img_cols
        Imag = np.ndarray((x2[-1], y2[-1], 1), dtype=np.uint8)
        # coordinate pairs on loaded image
        lx1 = np.ones(x_num, dtype=np.int32) * self.img_rows // 4
        lx2 = np.ones(x_num, dtype=np.int32) * self.img_rows // 4 * 3
        ly1 = np.ones(y_num, dtype=np.int32) * self.img_cols // 4
        ly2 = np.ones(y_num, dtype=np.int32) * self.img_cols // 4 * 3
        lx1[0] = 0
        ly1[0] = 0
        lx2[-1] = self.img_rows
        ly2[-1] = self.img_cols
        assert r_step == c_step  # FIXME: here can be generalized to different r, c
        # stich the tiles
        for i in range(imgs.shape[0]):  # assume the images are not
            # id of image patch
            ix = (i % (x_num * y_num)) // y_num
            iy = (i % (x_num * y_num)) % y_num
            img = imgs[i]
            Imag[x1[ix]:x2[ix], y1[iy]:y2[iy], 0] = img[lx1[ix]:lx2[ix], ly1[iy]:ly2[iy], 0] * 255
            if (i + 1) % (x_num * y_num) == 0:
                result = array_to_img(Imag)
                result.save(join(self.proj_dir, "LGN_train/trainborders/test_%04d.png" % (i // (x_num * y_num))))
                Imag = np.ndarray((x2[-1], y2[-1], 1), dtype=np.uint8)


if __name__ == '__main__':
    myunet = myUnet(img_rows=128, img_cols=128)
    # network train
    myunet.train()
    # network inference
    myunet.load_trained_model()
    myunet.save_img()
