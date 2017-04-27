#!env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
from six.moves import xrange
import tensorflow as tf
import svhn_input as si
import numpy as np
import time,sys
from datetime import datetime

BATCH_SIZE = 10
MAD = 0.9999
NUM_SAMPLES_DECAY = 350.0
LR_DECAY_FACTOR = 0.1
LR_INIT = 0.1
NAME_GPU = "tower"
NUME_STEPS = 2000
SAVE_FILE = si.DATA_DIR + "/model.ckpt"

def _activation_summary(v):
    tensor_name = re.sub('%s_[0-9]*/'%NAME_GPU,'',v.op.name)
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

def input_placsholders(batch_size):

    images_placeholder = tf.placeholder(dtype = tf.float32, shape = (batch_size, 32, 32, 3))
    labels_placeholder = tf.placeholder(dtype = tf.uint8, shape = (batch_size))
    return images_placeholder, labels_placeholder

def feed_data(data_set, batch_size, image_placeholder, label_placeholder):
    image_feed, label_feed = data_set.next_batch(batch_size)

    return {
            image_placeholder:image_feed,
            label_placeholder:label_feed
            }

def inference(images):

    with tf.variable_scope("conv1") as scope:
        kernel = _variable_with_weights_decay("weights",
                                              shape = [5,5,3,64],
                                              stddev = 5e-2,
                                              decay = 0.0,
                                              )

        conv = tf.nn.conv2d(images, kernel,strides=[1,1,1,1],padding='SAME')
        biases = _variable_on_cpu('biases', [64], init=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

    pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool1')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    with tf.variable_scope("conv2") as scope:
        kernel = _variable_with_weights_decay("weights",
                                              shape = [5,5,64,64],
                                              stddev = 5e-2,
                                              decay = 0.0,
                                              )

        conv = tf.nn.conv2d(norm1, kernel, strides=[1,1,1,1],padding='SAME')
        biases = _variable_on_cpu('biases', [64], init=tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)

    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    pool2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool2')

    with tf.variable_scope("local3") as scope:
        shape = pool2.get_shape().as_list()
        fc = tf.reshape(pool2, [shape[0], -1])
        dim = fc.get_shape()[1].value
        weights = _variable_with_weights_decay("weights", [dim, 384], stddev=0.04, decay=0.004)
        biases = _variable_on_cpu("biases", [384], init=tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(tf.matmul(fc, weights), biases)
        local3 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(local3)

    with tf.variable_scope("local4") as scope:
        weights = _variable_with_weights_decay("weights", [384, 192], stddev=0.04, decay=0.004)
        biases = _variable_on_cpu("biases", [192], init=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)

    with tf.variable_scope("softmax_liner") as scope:
        weights = _variable_with_weights_decay("weights", [192,10], stddev=1/192.0,decay=0.0)
        biases = _variable_on_cpu("biases",[10], init=tf.constant_initializer(0.0))
        softmax_liner = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(softmax_liner)

    return softmax_liner

def loss(logits, labels):
    labels_ = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels_,
                                                                   name="x_entropy_per_example")
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name="x_entropy_mean")
    tf.add_to_collection('losses',cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'),name="total_loss")

def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op

def evaluation(logits, labels):
    labels = tf.cast(labels, tf.int32)
    correct_op = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct_op, tf.int32))

def do_eval(sess, eval_op, images_placeholder, labels_placeholder, data_set, batch_size):
    true_count = 0
    step_per_epoch = data_set.num_examples
    num_examples = step_per_epoch * batch_size

    for sp in xrange(step_per_epoch):
        feed_dict = feed_data(data_set, batch_size, images_placeholder, labels_placeholder)
        true_count += sess.run(eval_op, feed_dict = feed_dict)
    precision = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))


def train(total_loss, global_step):

    lr = tf.train.exponential_decay(LR_INIT,
                                    global_step,
                                    int(si.NUM_PER_EPOCH_FOR_TRAIN / BATCH_SIZE * NUM_SAMPLES_DECAY),
                                    LR_DECAY_FACTOR,
                                    staircase=True,
                                    )

    tf.summary.scalar('learning_rate',lr)

    loss_averages_op = _add_loss_summaries(total_loss)
    with tf.control_dependencies([loss_averages_op]):
        with tf.name_scope('train') as scope:
            optimize = tf.train.GradientDescentOptimizer(lr).minimize(total_loss,global_step=global_step)

    return optimize,lr

def do_train():
    train_data = si.read_data_sets()

    global_step = tf.contrib.framework.get_or_create_global_step()

    # uint8images, labels = train_data.train.next_batch(10)
    # images = si.per_image_standardization(uint8images)

    images, labels = input_placsholders(BATCH_SIZE)

    logits = inference(images)
    total_loss = loss(logits, labels)
    train_op,learning_rate = train(total_loss, global_step)
    eval_op = evaluation(logits, labels)

    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()

    start_time = time.time()
    with tf.Session() as sess:

        #saver.restore(sess, SAVE_FILE)
        #print("model restored!")

        sess.run(init_op)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        merged = tf.summary.merge_all()
        run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        for step in xrange(NUME_STEPS):
            _feed_dict = feed_data(train_data.train, BATCH_SIZE, images, labels)

            _, l, lrr = sess.run([train_op, total_loss, learning_rate],feed_dict=_feed_dict)

            duration = time.time() - start_time

            if step % 10 == 0:
                elapsed_time = time.time() - start_time
                start_time = time.time()
                examples_per_sec = BATCH_SIZE / duration
                format_str = ('step %d, loss = %.2f  learning rate = %.6f  (%.1f examples/sec; %.2f ''sec/batch)')
                print(format_str % (step, l, lrr, examples_per_sec, duration))

                sys.stdout.flush()

            if step % 1000 == 0 or (step + 1) == NUME_STEPS:
                print("Traning Data Eval:")
                #do_eval(sess, eval_op, images, labels, train_data.train, BATCH_SIZE)
                print("Validation Data Eval:")
                #do_eval(sess, eval_op, images, labels, train_data.validate, BATCH_SIZE)
                print("Test Data Eval:")
                do_eval(sess, eval_op, images, labels, train_data.test, BATCH_SIZE)


        save_path = saver.save(sess, SAVE_FILE)
        print("model saved in file %s"%save_path)

def moving_average_train(total_loss, global_step):
    lr = tf.train.exponential_decay(LR_INIT,
                                    global_step,
                                    int(si.NUM_PER_EPOCH_FOR_TRAIN / BATCH_SIZE * NUM_SAMPLES_DECAY),
                                    LR_DECAY_FACTOR,
                                    staircase=True,
                                    )

    tf.summary.scalar('learning_rate', lr)

    loss_averages_op = _add_loss_summaries(total_loss)
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    for grad,var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name+'/gradients', grad)

    variable_averages = tf.train.ExponentialMovingAverage(MAD, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op

def do_ma_train():
    with tf.Graph().as_default():
        train_data = si.read_data_sets()
        global_step = tf.contrib.framework.get_or_create_global_step()

        images, labels = input_placsholders(BATCH_SIZE)
        logits = inference(images)
        total_loss = loss(logits, labels)
        train_op  = moving_average_train(total_loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):

            def begin(self):
                self._step = -1

            def before_run(self,run_context):
                self._step += 1
                self._start_time = time.time()
                return tf.train.SessionRunArgs(total_loss)

            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time
                loss_value = run_values.results
                if self._step % 10 == 0:
                    num_examples_per_step = BATCH_SIZE
                    example_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        example_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
            checkpoint_dir = si.DATA_DIR + "/train",
            hooks=[tf.train.StopAtStepHook(last_step=NUME_STEPS),
                   tf.train.NanTensorHook(total_loss),
                    _LoggerHook(),
                   ],
        ) as mon_sess:
            while not mon_sess.should_stop():
                _feed_dict = feed_data(train_data.train, BATCH_SIZE, images, labels)
                mon_sess.run([train_op],feed_dict=_feed_dict)


if __name__ == '__main__':
    if(len(sys.argv)) > 1:
        print("svhn_fast will run at level:", sys.argv[1])
        level = int(sys.argv[1])
        if level == 0:
            do_train()
        else:
            do_ma_train()
    else:
        print("please specify the run level include (0,1,2)")












































