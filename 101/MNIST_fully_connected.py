#!env python
#author:jingmcay@gmail.com

import tensorflow as tf
import MNIST_model as mnist
from six.moves import xrange
import input_data
import time


def input_placsholders(batch_size):

    images_placeholder = tf.placeholder(dtype = tf.float32, shape = (batch_size, mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(dtype = tf.int32, shape = (batch_size))

    return images_placeholder, labels_placeholder

def feed_fill_dict(data_set, batch_size, image_pl, label_pl):
    images_feed, labels_feed = data_set.next_batch(batch_size)
    feed_dict = {
        image_pl:images_feed,
        label_pl:labels_feed,
    }
    return feed_dict


def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set, batch_size):
    true_count = 0;
    steps_per_epoch = data_set.num_examples
    num_examples = steps_per_epoch * batch_size

    for sp in xrange(steps_per_epoch):
        feed_dict = feed_fill_dict(data_set,batch_size,images_placeholder,labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict = feed_dict)
    precision = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))


def do_training(max_step, batch_size, layer1_unit_size, layer2_unit_size, training_rate):

    data_set = input_data.read_data_sets("MNIST_data/")

    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = input_placsholders(batch_size)

        logist = mnist.predication(images_placeholder, layer1_unit_size, layer2_unit_size)

        loss = mnist.loss(logist, labels_placeholder)

        train_op = mnist.train(loss, training_rate)

        eval_correct = mnist.evaluation(logist, labels_placeholder)

        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter("MNIST_summary/", sess.graph)
            sess.run(init)

            for step in xrange(max_step):
                step_start_ts = time.time()

                feed_dict1 = feed_fill_dict(data_set.train, batch_size, images_placeholder, labels_placeholder)
                _,loss_value = sess.run([train_op, loss], feed_dict = feed_dict1)

                step_duration = time.time() - step_start_ts

                if step % 100 == 0:
                    # Print status to stdout.
                    print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, step_duration))
                    # Update the events file.
                    summary_str = sess.run(summary, feed_dict=feed_dict1)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()


                if (step + 1) % 1000 == 0 or (step + 1) == max_step:
                    print('Training Data Eval:')
                    do_eval(sess,
                            eval_correct,
                            images_placeholder,
                            labels_placeholder,
                            data_set.train,
                            batch_size)
                    # Evaluate against the validation set.
                    print('Validation Data Eval:')
                    do_eval(sess,
                            eval_correct,
                            images_placeholder,
                            labels_placeholder,
                            data_set.validation,
                            batch_size)
                    # Evaluate against the test set.
                    print('Test Data Eval:')
                    do_eval(sess,
                            eval_correct,
                            images_placeholder,
                            labels_placeholder,
                            data_set.test,
                            batch_size)



do_training(2000,100,128,32,0.01)





