"""
Specify the model for
"""

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
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, add, concatenate, Dense, Flatten, Lambda, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from tissue_classify.data_prep import pixel_classify_data_proc
from keras.utils import np_utils
#%%
# # Note merge is deprecated in Keras 2, add, concatenate is used
# def merge(input, mode='concat', concat_axis=3):
#     return concatenate(input, axis=concat_axis)

def dilate_pool(input, pool_size=(2,2), pooling_type="MAX", padding="VALID", dilation_rate=(1, 1), strides=(1,1)):
    """Since keras does not implement dilated pooling itself,
    Wrap around tensorflow dilated pooling function instead"""
    return tf.nn.pool(input, pool_size, pooling_type, padding, dilation_rate=dilation_rate, strides=strides)


class pixel_classifier_2d(object):

    def __init__(self, img_rows=65, img_cols=65,
                 proj_dir="/home/morganlab/Documents/ixP11LGN/TissueClassifier_Soma/",
                train_data_dir="/home/morganlab/Documents/ixP11LGN/TissueClassifier_Soma/"):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.proj_dir = proj_dir
        self.model_dir = join(proj_dir, 'Models/')
        self.train_data_dir = train_data_dir

    def load_traindata(self):
        processor = pixel_classify_data_proc(self.img_rows, self.img_cols, proj_dir=self.train_data_dir)
        imgs_train, labels_train = processor.load_train_data()
        return imgs_train, labels_train

    def load_testdata(self):
        processor = pixel_classify_data_proc(self.img_rows, self.img_cols, proj_dir=self.train_data_dir)
        imgs_test = processor.load_test_data()
        return imgs_test

    def get_net(self):
        '''define pixel classifier network structure with 2D input patches (Deprecated! )'''
        inputs = Input((self.img_rows, self.img_cols, 1))
        conv1 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(inputs)
        print("conv1 shape:", conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print("pool1 shape:", pool1.shape)
        pool1 = BatchNormalization()(pool1)

        conv2 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(pool1)
        print("conv2 shape:", conv2.shape)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print("pool2 shape:", pool2.shape)
        pool2 = BatchNormalization()(pool2)


        conv3 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(pool2)
        print("conv3 shape:", conv3.shape)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print("pool3 shape:", pool3.shape)
        pool3 = BatchNormalization()(pool3)

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

    def get_full_conv_net(self):
        '''define pixel classifier network structure with 2D input patches
        Use fully convolutional network for easy generalization to multi-pixel inference model
        '''
        inputs = Input((self.img_rows, self.img_cols, 1))
        conv1 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(inputs)
        print("conv1 shape:", conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print("pool1 shape:", pool1.shape)
        pool1 = BatchNormalization()(pool1)

        conv2 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(pool1)
        print("conv2 shape:", conv2.shape)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print("pool2 shape:", pool2.shape)
        pool2 = BatchNormalization()(pool2)

        conv3 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(pool2)
        print("conv3 shape:", conv3.shape)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print("pool3 shape:", pool3.shape)
        pool3 = BatchNormalization()(pool3)

        conv4 = Conv2D(16, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(pool3)
        print("conv4 shape:", conv4.shape)
        # conv4 = Flatten()(conv4)
        fc = Conv2D(16, 4, activation='relu', padding='valid', kernel_initializer='he_normal')(conv4)
        print("fc shape:", fc.shape)
        output = Conv2D(6, 1, activation='softmax', kernel_initializer='he_normal')(fc)  # by default softmax to -1 axis
        print("output shape:", output.shape)
        output = Flatten()(output)
        # output = Dense(units=6, activation='softmax')(fc)
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
        # imgs_test = self.load_testdata()
        print("loading data done")
        model = self.get_full_conv_net()
        print("got network")
        # checkponit to store the network parameter as .hdf5 file, change the name for different files saved
        pattern = "net_soma-{epoch:02d}-{val_acc:.2f}.hdf5"  # -{epoch:02d}-{val_acc:.2f}
        ckpt_path = join(self.model_dir, pattern)
        model_checkpoint = ModelCheckpoint(ckpt_path, monitor='loss', verbose=1, save_best_only=True)
        # model.load_weights('unet_LGN_mb.hdf5')
        print("Weight value loaded")
        print('Fitting model...')
        model.fit(imgs_train, onehot_labels, batch_size=16, epochs=20, verbose=1, validation_split=0.2, shuffle=True,
                  callbacks=[model_checkpoint])

    def train_generator(self, generator, valid_generator, **kwargs):
        """Alternative of train! Use generator to load sample during training
        More memory friendly
        """
        print("loading data")
        # imgs_test = self.load_testdata()
        print("loading data done")
        model = self.get_full_conv_net()
        print("got network")
        # checkponit to store the network parameter as .hdf5 file, change the name for different files saved
        pattern = "net_soma_ds-{epoch:02d}-{val_acc:.2f}.hdf5"  # -{epoch:02d}-{val_acc:.2f}
        ckpt_path = join(self.model_dir, pattern)
        model_checkpoint = ModelCheckpoint(ckpt_path, monitor='loss', verbose=1, save_best_only=True)
        # model.load_weights('unet_LGN_mb.hdf5')
        print("Weight value loaded")
        print('Fitting model...')
        model.fit_generator(generator, validation_data=valid_generator, epochs=20, verbose=1, shuffle=True,
                  callbacks=[model_checkpoint], **kwargs)

    # test the network with test dataset
    def load_trained_model(self, checkpoint_path=None):
        """Load checkpoint into model"""
        model = self.get_full_conv_net()
        print("got network")
        # load checkpoint to store the network parameter as .hdf5 file, change the name for different files saved
        if checkpoint_path==None:
            checkpoint_path = 'unet_LGN_mb.hdf5'
        model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True)
        model.load_weights(checkpoint_path)
        print('load trained model')
        # print("loading data")
        # imgs_test = self.load_testdata()
        # print("loading data done")
        # print('predict test data')
        # imgs_mask_test = model.predict(imgs_test, batch_size=5, verbose=1)
        # # create new folder to store the results
        # np.save(join(self.proj_dir, "LGN_train/trainborders/imgs_mask_test.npy"), imgs_mask_test)
        return model

    def get_inference_model(self, infer_rows=100, infer_cols=100):
        """ Transfer the weights learnt in the single output model into this patch output model
        The architecture translation rule between 2 model, see
        https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7493487
        EFFICIENT CONVOLUTIONAL NEURAL NETWORKS FOR PIXELWISE CLASSIFICATION ON HETEROGENEOUS HARDWARE SYSTEMS

        infer_rows, infer_cols can be arbitrarily large as long as GPU memory is enough
        """
        print("Getting inference model. ")
        inputs = Input((infer_rows, infer_cols, 1))
        conv1 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(inputs)
        print("conv1 shape:", conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')(conv1)
        print("pool1 shape:", pool1.shape)
        pool1 = BatchNormalization()(pool1)

        conv2 = Conv2D(64, 3, activation='relu', dilation_rate=(2, 2), padding='valid', kernel_initializer='he_normal')(pool1)
        print("conv2 shape:", conv2.shape)
        pool2 = Lambda(dilate_pool, arguments={'pool_size': (2, 2), 'dilation_rate': (2, 2), 'strides':(1,1),
                                               'padding': "VALID", })(conv2)
        # pool2 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(conv2) # No
        print("pool2 shape:", pool2.shape)
        pool2 = BatchNormalization()(pool2)

        conv3 = Conv2D(64, 3, activation='relu', dilation_rate=(4, 4), padding='valid', kernel_initializer='he_normal')(pool2)
        print("conv3 shape:", conv3.shape)
        pool3 = Lambda(dilate_pool, arguments={'pool_size': (2, 2), 'dilation_rate': (4, 4), 'strides': (1, 1),
                                               'padding': "VALID", })(conv3)
        # pool3 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1))(conv3)
        print("pool3 shape:", pool3.shape)
        pool3 = BatchNormalization()(pool3)

        conv4 = Conv2D(16, 3, activation='relu', dilation_rate=(8, 8), padding='valid', kernel_initializer='he_normal')(pool3)
        print("conv4 shape:", conv4.shape)
        # conv4 = Flatten()(conv4)
        fc = Conv2D(16, 4, activation='relu', dilation_rate=(8, 8), padding='valid', kernel_initializer='he_normal')(conv4)
        print("fc shape:", fc.shape)
        output = Conv2D(6, 1, activation='softmax', padding='valid', kernel_initializer='he_normal')(fc)
        # fc = Flatten()(fc)
        print("output shape:", output.shape)
        model = Model(input=inputs, output=output)

        model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy',
                      metrics=['accuracy'])  # , options = run_opts)
        # loss function `binary_crossentropy` or `categorical_crossentropy` here seems binary

        return model

    def transfer_weight_to_inference(self, ckpt_path, infer_rows=201, infer_cols=201):
        """ Transfer the weights learnt in the single output model into this patch output model"""
        model = self.load_trained_model(checkpoint_path=ckpt_path)
        inference_model = self.get_inference_model(infer_rows, infer_cols)
        inference_model.set_weights(model.get_weights())
        return inference_model

def inference_on_image(img, inference_model):
    '''General function to infer on whole image by pixel wise inference'''
    out_shape = inference_model.output_shape[1:-1]
    in_shape = inference_model.input_shape[1:-1]
    in_y, in_x = in_shape
    out_y, out_x = out_shape
    if (not (in_y - out_y) % 2 == 0) or (not (in_x - out_x) % 2 == 0):
        print("warning: margin is not integer")
    pady = np.int(np.ceil((in_y - out_y) / 2))
    padx = np.int(np.ceil((in_x - out_x) / 2))
    # pad the image
    pad_img = np.pad(img, ((pady, pady), (padx, padx)), 'reflect')
    size_y, size_x = pad_img.shape
    # get the image into stack
    img_stack = []
    coord_list = []

    step_y = out_y
    step_x = out_x
    csr_y = 0
    csr_x = 0
    x_reset_flag = False
    y_end_flag = False
    while True:
        # append
        img_stack.append(pad_img[csr_y:csr_y + in_y, csr_x:csr_x + in_x])
        coord_list.append((csr_y, csr_x))  # note no pad in output image!

        # moving decision
        if x_reset_flag and y_end_flag:
            break
        elif x_reset_flag:  # return
            csr_x = 0
            csr_y += step_y
            x_reset_flag = False
        else:
            csr_x += step_x

        # modify move
        if csr_x + in_x > size_x or csr_x + out_x > size_x - 2 * padx:
            csr_x = size_x - 2 * padx - out_x
            x_reset_flag = True
        if csr_y + in_y > size_y or csr_y + out_y > size_y - 2 * pady:
            csr_y = size_y - 2 * pady - out_y
            y_end_flag = True
    # send the stack to model for prediction
    img_stack = np.array(img_stack)
    img_stack.shape = img_stack.shape + (1,)
    y_prob_stack = inference_model.predict(img_stack)
    y_label_stack = np.squeeze(y_prob_stack.argmax(axis=-1))
    # reorder the tile images
    label_map = np.zeros(img.shape, dtype=np.uint8)
    for i, (csr_y, csr_x) in enumerate(coord_list):
        label_map[csr_y: csr_y + out_y, csr_x: csr_x + out_x] = y_label_stack[i, :, :]

    return label_map

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

#%%
if __name__ == '__main__':
    pc2 = pixel_classifier_2d(img_rows=65, img_cols=65,)
                              # train_data_dir="/scratch/binxu.wang/tissue_classifier",
                              # proj_dir="/scratch/binxu.wang/tissue_classifier")
    # network train
    pc2.train()

    inference_model = pc2.get_inference_model()
    # network inference
    # pc2.load_trained_model()
    # pc2.save_img()
    model = pc2.load_trained_model("/Users/binxu/Connectomics_Code/tissue_classifier/Models/net_soma-01-0.89.hdf5")
