#!env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import tensorflow as tf
import svhn_input as si
import svhn_fast

def per_image_standardization(images):
    _images = []

    for i in images:
        img = tf.cast(i, tf.float32)
        img = tf.image.per_image_standardization(img)
        _images.append(img)

    return _images


train_data = si.read_data_sets()

global_step = tf.contrib.framework.get_or_create_global_step()

images,labels = train_data.train.next_batch(10)

# images = per_image_standardization(images)


logits = svhn_fast.inference(images)
loss = svhn_fast.loss(logits,labels)
train_op = svhn_fast.train(loss, global_step)


eval_op = svhn_fast.evaluation(logits, labels)
# eval_op = tf.no_op(name='123_')
arg_op = tf.nn.in_top_k(logits, tf.cast(labels, tf.int32), 1)
aa_op = tf.reduce_sum(tf.cast(arg_op, tf.int32))

init = tf.global_variables_initializer()



with tf.Session() as sess:
    sess.run(init)
    opt,logits_value,loss_value,aop,bop = sess.run([train_op,logits, loss,aa_op,arg_op])
    print(logits_value)
    print(labels)
    print(bop)
    print(aop)


