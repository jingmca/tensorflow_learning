#!env python
#author: jingmcay@gmail.com

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys

from six.moves import urllib
import tensorflow as tf

import cifar_cnn_input

BATCH_SIZE = 128
DATA_DIR = cifar_cnn_input.DATA_DIR
IMAGE_SIZE = cifar_cnn_input.IMAGE_SIZE
NUM_EXAMPLE_PER_TRAIN = cifar_cnn_input.NUM_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLE_PER_EVAL = cifar_cnn_input.NUM_PER_EPOCH_FOR_EVAL

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

TOWER_NAME = "tow"

def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))

def _variable_on_cpu(x, shape, init):
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(x,shape,dtype=dtype,initializer=init)

    return var

def _variable_with_weight_decay(x,shape, stddev, wd):
    var = _variable_on_cpu(x, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var),wd,name="weight_loss")
        tf.add_to_collection('losses',weight_decay)
    return var

def inference(images):

    #conv1
    with tf.variable_scope("conv1") as scope:
        kernel = _variable_with_weight_decay('weights',[5,5,3,64],stddev=5e-2,wd=0.0)
        conv1 = tf.nn.conv2d(images,kernel,strides=[1,1,1,1],padding="SAME")
        biases = _variable_on_cpu("biases",[64],tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv1,biases)
        conv1 = tf.nn.relu(pre_activation,name=scope.name)
        _activation_summary(conv1)

    #pool1
    pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name="pool1")

    #norm1
    norm1 = tf.nn.lrn(pool1,4,bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

    #conv2
    with tf.variable_scope("conv2") as scope:
        kernel = _variable_with_weight_decay('weights',[5,5,64,64],stddev=5e-2,wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, strides=[1,1,1,1],padding='SAME')
        biases = _variable_on_cpu('biases',[64],tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation,name=scope.name)
        _activation_summary(conv2)

    #norm2
    norm2 = tf.nn.lrn(conv2, 4, bias = 1.0, alpha=0.001/9.0, beta=0.75,name='norm2')

    #pool2
    pool2 = tf.nn.max_pool(norm2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name="pool2")

    #local3
    with tf.variable_scope("local3") as scope:
        reshape = tf.reshape(pool2,[BATCH_SIZE,-1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weight',[dim,384],stddev=0.4,wd=0.004)
        biases = _variable_on_cpu('biases',[384],tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape,weights) + biases, name=scope.name)
        _activation_summary(local3)

    #local4
    with tf.variable_scope("local4") as scope:
        weights = _variable_with_weight_decay('weight',[384,192],stddev=0.4,wd=0.04)
        biases = _variable_on_cpu('biases',[192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3,weights) + biases,name=scope.name)
        _activation_summary(local4)

    #sofmax liner layer
    with tf.variable_scope("softmaxliner") as scope:
        weights = _variable_with_weight_decay('weight',[192,10],stddev=0.1/192,wd=0.0)
        biases = _variable_on_cpu('biases',[10],tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4,weights) ,biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear

def losses(logits,labels):

    labels = tf.cast(labels,tf.int64)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name="cross_entropy_per_example")
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy_mean')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection("losses"),name="total_loss")

def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op

def train(total_loss,global_step):
    num_batches_per_epoch = NUM_EXAMPLE_PER_TRAIN / BATCH_SIZE
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,global_step=global_step,decay_steps=decay_steps,decay_rate=LEARNING_RATE_DECAY_FACTOR,staircase=True)
    tf.summary.scalar('learning_rate', lr)

    loss_averages_op = _add_loss_summaries(total_loss)
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grad = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grad, global_step=global_step)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad_, var in grad:
        if grad_ is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad_)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op



















