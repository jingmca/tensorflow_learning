#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector


LOG_DIR = "tmp/"
DATA_DIR = LOG_DIR + "MNIST_DATA/"
LEARNING_RATE = 0.001
MAX_STEPS = 1000
DROPOUT = 0.9

def train():
    mnist = input_data.read_data_sets(DATA_DIR,one_hot=True)

    sess  = tf.InteractiveSession()

    #create nn layer
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None,10], name='y-input')

    with tf.name_scope("input_reshape"):
        image_reshape_input = tf.reshape(x,[-1,28,28,1])
        tf.summary.image("input",image_reshape_input,10)

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1,shape=shape)
        return tf.Variable(initial)

    def variable_summary(var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max',tf.reduce_max(var))
        tf.summary.scalar('min',tf.reduce_min(var))
        tf.summary.histogram('histogram',var)

    def nn_layer(input_tensor, input_dim, output_dim, layer_name, act = tf.nn.relu):
        with tf.name_scope("weights"):
            weights = weight_variable([input_dim,output_dim])
            variable_summary(weights)
        with tf.name_scope("biases"):
            biases = bias_variable([output_dim])
            variable_summary(biases)
        with tf.name_scope("Wx_plus_b"):
            p = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('Wx_plus_b', p)
        with tf.name_scope("activation"):
            activation = act(p, name='activation')
            tf.summary.histogram('activation', activation)
        return activation

    #buildup the NN
    hidden1 = nn_layer(x, 784, 500, 'layer1')

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(hidden1,keep_prob)

    y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)

    tf.summary.scalar('cross_entropy',cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.summary.scalar('accuracy',accuracy)

    merged = tf.summary.merge_all()
    train_w = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)
    test_w = tf.summary.FileWriter(LOG_DIR + '/test')
    tf.global_variables_initializer().run()

    def feed_dict(train):
        if train:
            xs, ys = mnist.train.next_batch(100)
            k = DROPOUT
        else:
            xs,ys = mnist.test.images, mnist.test.labels
            k = 1.0
        return {x:xs,y_:ys,keep_prob:k}

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(LOG_DIR)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = y.name

    projector.visualize_embeddings(summary_writer, config)
    embedding.sprite.image_path = LOG_DIR + "mnist_10k_sprite.png"
    embedding.sprite.single_image_dim.extend([28, 28])

    for i in range(MAX_STEPS):
        saver.save(sess, LOG_DIR + "model.ckpt", i)

        if i % 10 == 0:  # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_w.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:  # Record train set summaries, and train
            if i % 100 == 99:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_w.add_run_metadata(run_metadata, 'step%03d' % i)
                train_w.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:  # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_w.add_summary(summary, i)
    train_w.close()
    test_w.close()



train()













