#!env python
#author: jingmcay@gmail.com

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

#classification number
NUM_CLASSES = 10

#image size and pixels
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def weight_init(inputs, dims):
    return tf.Variable(tf.truncated_normal([inputs, dims], stddev = 1.0 / math.sqrt(float(inputs))), name= 'weights')

def biase_init(dims):
    return tf.Variable(tf.zeros([dims], dtype = tf.float32), name = 'biases')

def predication(inputs, layer1_units, layer2_units):
    """build the 3-layer NN model for MNIST
    """

    #Layer1
    with tf.name_scope('layer1'):
        weights = weight_init(IMAGE_PIXELS, layer1_units)
        biases = biase_init(layer1_units)
        layer1 = tf.nn.relu(tf.matmul(inputs, weights) + biases)

    #layer2
    with tf.name_scope('layer2'):
        weights = weight_init(layer1_units, layer2_units)
        biases = biase_init(layer2_units)
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)

    #liner-layer
    with tf.name_scope('liner'):
        weights = weight_init(layer2_units,NUM_CLASSES)
        biases = biase_init(NUM_CLASSES)
        logist = tf.matmul(layer2,weights) + biases

    return logist


def loss(logist, labels):
    """calculate the cross-entropy with the model output and labels
    """
    labels_ = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logist, labels_, name = 'xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def train(loss, learning_rate):
    """train this model with a specific learning rate
    """
    tf.summary.scalar('loss', loss)

    global_step = tf.Variable(0, name = 'gloal_step', trainable = False)
    training = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss=loss, global_step=global_step, name='training_phrase')

    return training

def evaluation(logist, labels):
    correct = tf.nn.in_top_k(logist, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))







