#!env python
#author jingmcay@gmail.com

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange

import tensorflow as tf

IMAGE_SIZE = 24
NUM_PER_EPOCH_FOR_TRAIN = 50000
NUM_PER_EPOCH_FOR_EVAL = 10000
NUM_PRE_PROCESS_THREADS = 16
DATA_DIR = os.path.curdir + "/cifar-10-batches-bin/"
EVAL_DIR = os.path.curdir

def decode_cifar(fnq):

    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()

    label_bytes = 1
    result.height, result.weight = 32,32
    result.depth = 3
    image_bytes = result.height * result.weight * result.depth
    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes = record_bytes)
    result.key, value = reader.read(fnq)

    record_bytes = tf.decode_raw(value, tf.uint8)
    result.label = tf.cast(tf.strided_slice(record_bytes,[0],[label_bytes],[1]), tf.int32)
    depth_image_bytes = tf.reshape(tf.strided_slice(record_bytes,[label_bytes],[label_bytes + image_bytes],[1]),[result.depth, result.height, result.weight])

    result.uint8image = tf.transpose(depth_image_bytes,[1,2,0])

    return result

def _gen_images_and_labels_batch(image, label, min_q_example, batch_size, shuffle):
    if shuffle:
        images,labels = tf.train.shuffle_batch([image,label],batch_size,capacity=min_q_example + 3*batch_size,min_after_dequeue=min_q_example,num_threads=NUM_PRE_PROCESS_THREADS)
    else:
        images,labels = tf.train.batch([image,label],batch_size,num_threads=NUM_PRE_PROCESS_THREADS,capacity=min_q_example + 3*batch_size)
    tf.summary.image('images',images)
    return  images,tf.reshape(labels, [batch_size])

def random_inputs(batch_size):
    filenames = [os.path.join( DATA_DIR + "data_batch_%d.bin"%i) for i in xrange(1,6)]

    fnq = tf.train.string_input_producer(filenames)
    read_input = decode_cifar(fnq)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = weight = IMAGE_SIZE

    random_corp_image = tf.random_crop(reshaped_image,[height,weight,3])
    random_corp_image = tf.image.random_flip_left_right(random_corp_image)
    random_corp_image = tf.image.random_brightness(random_corp_image,max_delta=63)
    random_corp_image = tf.image.random_contrast(random_corp_image,lower=0.2,upper=1.3)
    float_image = tf.image.per_image_standardization(random_corp_image)

    float_image.set_shape([height,weight,3])
    read_input.label.set_shape([1])

    min_fraction_image_in_queue = 0.4
    min_queue_example = int(NUM_PER_EPOCH_FOR_TRAIN * min_fraction_image_in_queue)

    return  _gen_images_and_labels_batch(float_image,read_input.label,min_queue_example,batch_size,shuffle=True)

def inputs(eval_data,batch_size):
    if not eval_data:
        filenames = [os.path.join(DATA_DIR + "data_batch_%d.bin"%i) for i in xrange(1,6)]
        num_examples_ecpho_queue = NUM_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(DATA_DIR, "test_batch.bin")]
        num_examples_ecpho_queue = NUM_PER_EPOCH_FOR_EVAL

    fnq = tf.train.string_input_producer(filenames)
    read_input = decode_cifar(fnq)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = weight = IMAGE_SIZE
    corp_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,height,weight)
    float_image = tf.image.per_image_standardization(corp_image)

    float_image.set_shape([height,weight,3])
    read_input.label.set_shape([1])

    min_fraction_image_in_queue = 0.4
    min_queue_example = int(NUM_PER_EPOCH_FOR_TRAIN * min_fraction_image_in_queue)

    return _gen_images_and_labels_batch(float_image,read_input.label,min_queue_example,batch_size,shuffle=False)