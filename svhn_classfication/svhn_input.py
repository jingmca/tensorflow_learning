#!env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange
import scipy as sp
import scipy.io as sio
from scipy.misc import *
import numpy as np

import tensorflow as tf

IMAGE_SIZE = 24
NUM_PER_EPOCH_FOR_VALI = 3000
NUM_PER_EPOCH_FOR_TRAIN = 73257 - NUM_PER_EPOCH_FOR_VALI
NUM_PER_EPOCH_FOR_EVAL = 26032
NUM_PRE_PROCESS_THREADS = 16
NUM_CLASS = 10
DATA_DIR = os.path.curdir + "/svhn_data"
EVAL_DIR = os.path.curdir

def _read_mat_test():
    train_file = DATA_DIR + "/train_32x32.mat"
    train_dict = sio.loadmat(train_file)
    X_ = np.asarray(train_dict['X'])
    train_data_y = train_dict['y']

    train_data_x = []
    for i in range(X_.shape[3]):
        train_data_x.append(X_[:,:,:,i])
    train_data_x = np.asarray(train_data_x)

    for j in range(len(train_data_y)):
        if(train_data_y[j] % 10 == 0):
            train_data_y[j] = 0

    return (train_data_x, train_data_y)

def read_svhn_mat(filename, shape):
    train_dict = sio.loadmat(os.path.join(DATA_DIR,filename))
    X_ = np.asarray(train_dict['X'])
    train_data_y = train_dict['y']

    train_data_x = []
    for i in range(X_.shape[3]):
        #train_data_x.append(X_[:, :, :, i])
        img_ = X_[:,:,:,i]
        # img_ = tf.cast(img_, tf.float32)
        #img_ = img_.astype(np.float32)
        #img_ = tf.image.per_image_standardization(img_)
        #
        img_ = img_.astype(np.float32)
        img_ = np.multiply(img_, 1.0 / 255.0)
        train_data_x.append(img_)

    train_data_x = np.asarray(train_data_x)

    for j in range(len(train_data_y)):
        if (train_data_y[j] % 10 == 0):
            train_data_y[j] = 0
    train_data_y = np.asarray(train_data_y).reshape(shape)

    return (train_data_x, train_data_y)

def train_data():
    return read_svhn_mat("train_32x32.mat", shape = [NUM_PER_EPOCH_FOR_TRAIN + NUM_PER_EPOCH_FOR_VALI])

def test_data():
    return read_svhn_mat("test_32x32.mat", shape = [NUM_PER_EPOCH_FOR_EVAL])

def _generate_image_and_label_batch(image,label, batch_size, shuffle = True):
    min_after_dequeue = int(0.1 * NUM_PER_EPOCH_FOR_TRAIN)
    num_preprocess_threads = 4

    if shuffle:
        image_batch, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity= min_after_dequeue + 3 * batch_size,
            min_after_dequeue= min_after_dequeue)
    else:
        image_batch, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_after_dequeue + 3 * batch_size,
            )

    tf.summary.image('images', image_batch)

    print(image_batch)
    print(label_batch)

    return image_batch, tf.reshape(label_batch, [batch_size])

class Input_Data(object):

    def __init__(self, images, labels, one_hot = False):

        assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape,
                                                 labels.shape)
                                                    )
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0


    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):

        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if(self._index_in_epoch > self._num_examples):
            self._epochs_completed += 1
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end]

def read_data_sets():
    class DataSets(object):
        pass
    data_sets = DataSets()

    train_images, train_labels = train_data()

    vali_images = train_images[-(NUM_PER_EPOCH_FOR_VALI):]
    vali_labels = train_labels[-(NUM_PER_EPOCH_FOR_VALI):]

    test_images, test_labels = test_data()

    data_sets.train = Input_Data(train_images[:NUM_PER_EPOCH_FOR_TRAIN],train_labels[:NUM_PER_EPOCH_FOR_TRAIN])
    data_sets.validate = Input_Data(vali_images, vali_labels)
    data_sets.test = Input_Data(test_images, test_labels)
    print(data_sets.train._index_in_epoch)
    return data_sets

def per_image_standardization(images):
    _images = []

    for i in images:
        img = tf.cast(i, tf.float32)
        img = tf.image.per_image_standardization(img)
        _images.append(img)

    return _images

#
# x,y = train_data()
# print(x)
# i = Input_Data(data = y,label=True)
# for k in range(4):
#     print(i.batch_read(5))
#     print("......")




if __name__ == '__main__':
    # x,y = train_data()
    # print (x)
    # y = y.reshape([NUM_PER_EPOCH_FOR_TRAIN])
    # indexdY = [[i, y[i]] for i in range(len(y))]
    # y = tf.one_hot(y,10)
    # # print(indexdY)
    # # y = tf.sparse_to_dense(indexdY,(NUM_PER_EPOCH_FOR_TRAIN,NUM_CLASS),1.0,0.0)
    #
    # with tf.Session() as sess:
    #     y_ = sess.run(y)
    #     print(y_.shape)
    a = read_data_sets()
    val = a.test
    print(val)
    images, labels = val.next_batch(10)
    print(images)
    img0 = images[0]

    print(img0)

    # img0 = img0.astype(np.float32)
    # img0 = np.multiply(img0, 1.0 / 255.0)
    # print(img0)

    # img0 = tf.cast(img0, tf.float32)
    # img0 = tf.image.per_image_standardization(img0)