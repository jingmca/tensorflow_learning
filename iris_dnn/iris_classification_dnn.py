#!env python
#authur:jingmcay@gmail.com

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

#date set
IRIS_TRAINING_SET = "iris_training.csv"
IRIS_TEST_SET = "iris_test.csv"

traning_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename = IRIS_TRAINING_SET,
    target_dtype = np.int,
    features_dtype = np.float32
)

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename = IRIS_TEST_SET,
    target_dtype = np.int,
    features_dtype = np.float32
)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension = 4)]

dnn_classifier = tf.contrib.learn.DNNClassifier(feature_columns = feature_columns,
                                                hidden_units = [10,20,10],
                                                n_classes = 3,
                                                model_dir = "iris_model")

dnn_classifier.fit(x = traning_set.data,
                   y = traning_set.target,
                   steps = 2000,)

accuracy = dnn_classifier.evaluate(x = test_set.data,
                                   y = test_set.target)["accuracy"]

print('Accuracy: {0:f}'.format(accuracy))

new_objects = np.array(
    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)

y = list(dnn_classifier.predict(new_objects, as_iterable = True))
print('Predictions: {}'.format(str(y)))