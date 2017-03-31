#!env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import tensorflow as tf
import cifar_cnn_input

class trainObject:
    pass

train_object = trainObject()
train_object.batch_size = 128
train_object.data_dir = cifar_cnn_input.DATA_DIR
train_object.corp_size = cifar_cnn_input.IMAGE_SIZE
train_object.num_samples_per_train = cifar_cnn_input.NUM_PER_EPOCH_FOR_TRAIN
train_object.num_samples_per_eval = cifar_cnn_input.NUM_PER_EPOCH_FOR_EVAL
train_object.MAD = 0.9999
train_object.num_samples_decay = 350.0
train_object.learn_decay_factor = 0.1
train_object.init_learning_rate = 0.1
train_object.multi_gpu_tower = "tower"

def _activation_summary(v):
    tensor_name = re.sub('%s_[0-9]*/'%train_object.multi_gpu_tower,'',v.op.name)
    tf.summary.histogram(tensor_name+'/activations',v)
    tf.summary.scalar(tensor_name+'/sparsity', tf.nn.zero_fraction(v))
    print(v)

def _variable_on_cpu(v, shape, init):
    with tf.device('/cpu:0'):
        v_ = tf.get_variable(v,shape,dtype=tf.float32,initializer=init)
    return v_

def _variable_with_weights_decay(name, shape, stddev, decay):
    var_ = _variable_on_cpu(name, shape, init=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    if decay is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var_), decay, "weight_loss")
        tf.add_to_collection("losses",weight_decay)
    return var_


def inference(images):
    """
    eight layer for NN
    """
    with tf.variable_scope("conv1") as scope:
        kernel = _variable_with_weights_decay("weights",
                                              shape = [5,5,3,64],
                                              stddev = 5e-2,
                                              decay = 0.0,
                                              )
        conv_ = tf.nn.conv2d(images, kernel, [1,1,1,1], padding = 'SAME')
        biases_ = _variable_on_cpu("biases", [64], init = tf.constant_initializer(0.0))
        conv1 = tf.nn.relu(tf.nn.bias_add(conv_, biases_), name = scope.name)
        _activation_summary(conv1)

    pool1 = tf.nn.max_pool(conv1, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'SAME', name = "pool1")

    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    with tf.variable_scope("conv2") as scope:
        kernel = _variable_with_weights_decay("weights",
                                              shape = [5,5,64,64],
                                              stddev = 5e-2,
                                              decay = 0.0
                                              )
        conv__ = tf.nn.conv2d(norm1, kernel, strides = [1,1,1,1], padding = 'SAME')
        biases__ = _variable_on_cpu("biases", [64], init = tf.constant_initializer(0.1))
        conv2 = tf.nn.relu(tf.nn.bias_add(conv__, biases__), name = scope.name)
        _activation_summary(conv2)

    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')

    pool2 = tf.nn.max_pool(norm2, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'SAME', name = "pool2")

    with tf.variable_scope("local3") as scope:
        reshape = tf.reshape(pool2, [train_object.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weights_decay("weights",
                                               shape = [dim, 384],
                                               stddev = 0.04,
                                               decay = 0.004
                                               )
        biases___ = _variable_on_cpu("biases", [384], init = tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases___, name = scope.name)
        _activation_summary(local3)

    with tf.variable_scope("local4") as scope:
        weights_ = _variable_with_weights_decay("weights",
                                                shape = [384,192],
                                                stddev = 0.04,
                                                decay = 0.004)
        biases____ = _variable_on_cpu("biases", [192], init = tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights_) + biases____, name = scope.name)
        _activation_summary(local4)

    with tf.variable_scope("softmax_liner") as scope:
        weights__ = _variable_with_weights_decay("weights",
                                                shape = [192, 10],
                                                stddev = 1 / 192.0,
                                                decay = 0.0
                                                )
        biases_____ = _variable_on_cpu("biases", [10], init = tf.constant_initializer(0.0))
        softmax_liner = tf.add(tf.matmul(local4, weights__), biases_____, name = scope.name)
        _activation_summary(softmax_liner)

    return softmax_liner



def losses(logits,labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

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
    num_batches_per_epoch = train_object.num_samples_per_train  / train_object.batch_size
    decay_steps = int(num_batches_per_epoch * train_object.num_samples_decay)

    lr = tf.train.exponential_decay(train_object.init_learning_rate,global_step=global_step,decay_steps=decay_steps,decay_rate=train_object.learn_decay_factor,staircase=True)
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
        train_object.MAD, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op