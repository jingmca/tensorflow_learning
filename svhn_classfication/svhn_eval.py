#!env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
from six.moves import xrange
import tensorflow as tf
import svhn_input as si
import svhn_fast as fast
import numpy as np
import time,sys
import math
from datetime import datetime

eval_dir = si.DATA_DIR
eval_interval_secs = 5
num_examples = 1000
run_once = False

def eval_once(dataset, saver, summary_writer, top_k_op, summary_op, image_placeholder, label_placeholder):

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(eval_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print(global_step)
        else:
            print('No checkpoint file found!')
            return

        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
            num_iter = int(math.ceil(num_examples / fast.BATCH_SIZE))
            true_count = 0
            total_sample_count = num_iter * fast.BATCH_SIZE
            step = 0
            while step < num_iter and not coord.should_stop():
                feeds = fast.feed_data(dataset, fast.BATCH_SIZE, image_placeholder, label_placeholder)
                predicitions = sess.run([top_k_op],feed_dict=feeds)
                true_count += np.sum(predicitions)
                step += 1

            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
            print(global_step)

        except Exception as e:
            coord.request_stop(e)
        coord.request_stop()
        if threads:
            coord.join(threads, stop_grace_period_secs=2)

def evaluate():
    with tf.Graph().as_default() as g:
        train_data = si.read_data_sets()
        images, labels = fast.input_placsholders(fast.BATCH_SIZE)

        logits = fast.inference(images)
        labels = tf.cast(labels, tf.int32)
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # variable_averages = tf.train.ExponentialMovingAverage(
        #     fast.MAD)
        # variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(eval_dir, g)

        while True:
            eval_once(train_data.test, saver, summary_writer, top_k_op, summary_op, image_placeholder = images,
                      label_placeholder = labels)
            if run_once:
                break
            time.sleep(eval_interval_secs)



if __name__ == '__main__':
    evaluate()