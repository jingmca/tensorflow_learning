#!env python
#author: jingmcay@gmail.com

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]


def input_fn(df):
    continuous_cols = {k:tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}

    categorical_cols = {
                        k:tf.SparseTensor(
                            indices=[[i,0] for i in range(df[k].size)],
                            values = df[k].values,
                            dense_shape= [df[k].size, 1]
                        )
        for k in CATEGORICAL_COLUMNS
    }

    feature_cols = dict(continuous_cols.items() + categorical_cols.items())

    label = tf.constant(df[LABEL_COLUMN].values)

    return  feature_cols,label


def build_estimator(model_dir, model_type = "wide"):
    """feature selection & model buildup"""

    gender = tf.contrib.layers.sparse_column_with_keys(column_name = 'gender', keys = ['Male','Female'])
    education = tf.contrib.layers.sparse_column_with_hash_bucket("education", hash_bucket_size=1000)
    relationship = tf.contrib.layers.sparse_column_with_hash_bucket(
        "relationship", hash_bucket_size=100)
    workclass = tf.contrib.layers.sparse_column_with_hash_bucket("workclass", hash_bucket_size = 100)
    occupation = tf.contrib.layers.sparse_column_with_hash_bucket("occupation", hash_bucket_size = 1000)
    native_country = tf.contrib.layers.sparse_column_with_hash_bucket("native_country", hash_bucket_size = 1000)

    #Continuous base feature
    age = tf.contrib.layers.real_valued_column('age')
    education_num = tf.contrib.layers.real_valued_column('education_num')
    capital_gain = tf.contrib.layers.real_valued_column('capital_gain')
    capital_loss = tf.contrib.layers.real_valued_column('capital_loss')
    hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")

    #Bucketization feature
    age_bucket = tf.contrib.layers.bucketized_column(age, boundaries = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    #Cross feature
    education_x_occupation = tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size = int(1e4))
    age_bucket_x_education_x_occupation = tf.contrib.layers.crossed_column([age_bucket, education, occupation], hash_bucket_size = int(1e6))

    #deep columns
    deep_columns = [tf.contrib.layers.embedding_column(workclass, dimension = 8),
                    tf.contrib.layers.embedding_column(education, dimension=8),
                    tf.contrib.layers.embedding_column(gender, dimension=8),
                    tf.contrib.layers.embedding_column(relationship, dimension=8),
                    tf.contrib.layers.embedding_column(native_country,
                                                       dimension=8),
                    tf.contrib.layers.embedding_column(occupation, dimension=8),
                    age,
                    education_num,
                    capital_gain,
                    capital_loss,
                    hours_per_week,
                    ]

    #estimator buildup
    if model_type == "wide":
        m = tf.contrib.learn.LinearClassifier(feature_columns = [gender, native_country, education, occupation, workclass,
  age_bucket, education_x_occupation, age_bucket_x_education_x_occupation],
                                              optimizer = tf.train.FtrlOptimizer(
                                                  learning_rate = 0.1,
                                                  l1_regularization_strength = 1.0,
                                                  l2_regularization_strength = 1.0,
                                              ),
                                              model_dir = model_dir)
    elif model_type == "wide_n_deep":
        m = tf.contrib.learn.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns = [gender, native_country, education, occupation, workclass,
  age_bucket, education_x_occupation, age_bucket_x_education_x_occupation],
            dnn_feature_columns = deep_columns,
            dnn_hidden_units=[100, 50])

    else:
        m = None

    return m








