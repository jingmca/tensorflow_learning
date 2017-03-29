#!env python

import cifar_cnn_input as input
import os
import tensorflow as tf

def gen_sample_queue():
    filenames = [os.path.join(os.path.curdir,"data_batch_%d.bin"%i) for i in range(5)]
    print filenames
    filename_queue = tf.train.string_input_producer(filenames)

    result  = input.decode_cifar(filename_queue)
    return result.uint8image


gen_sample_queue()