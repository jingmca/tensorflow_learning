#!env python
#author: jingmcay@gmail.com

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def model(features, labels, mode):
    W = tf.get_variable("W",[1],dtype=tf.float64)
    b = tf.get_variable("b",[1],dtype=tf.float64)
    y = W * features['x'] + b

    loss = tf.reduce_sum(tf.square( y - labels))
    global_steps = tf.train.get_global_step()
    opitimize = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(opitimize.minimize(loss),
                     tf.assign_add(global_steps,1))

    return tf.contrib.learn.ModelFnOps(mode = mode,
                                       predictions = y,
                                       loss = loss,
                                       train_op = train)


estimator = tf.contrib.learn.Estimator(model_fn = model)

x =np.array([1.,2.,3.,4.])
y =np.array([-1.,-2.,-3.,-4.])

input_fn = tf.contrib.learn.io.numpy_input_fn({'x':x}, y, 4, num_epochs = 1000)

estimator.fit(input_fn = input_fn, steps = 1000)
print(estimator.evaluate(input_fn = input_fn, steps = 10))