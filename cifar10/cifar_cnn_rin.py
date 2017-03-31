#!env python
#author jingmcay@gmail.com

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf
import cifar_cnn_model as model
import cifar_cnn_input
import cifar10_cnn

def train():
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        images,labels = cifar_cnn_input.random_inputs(model.BATCH_SIZE)

        logits = cifar10_cnn.inference(images)
        loss = cifar10_cnn.losses(logits,labels)
        train_op = cifar10_cnn.train(loss,global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1

            def before_run(self, run_context):
                self._step += 1
                self._start_time = time.time()
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time
                loss_value = run_values.results
                if self._step % 10 == 0:
                    num_examples_per_step = model.BATCH_SIZE
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=cifar_cnn_input.DATA_DIR,
                hooks=[tf.train.StopAtStepHook(last_step=1000000),
                       tf.train.NanTensorHook(loss),
                       _LoggerHook()],
                config=tf.ConfigProto(
                    log_device_placement=False)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run([train_op])


train()



