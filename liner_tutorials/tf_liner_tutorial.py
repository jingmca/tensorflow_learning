#!env python
#author jingmcay@gmail.com

import tf_liner_models as tflm
import sys
import tempfile

import pandas as pd
import tensorflow as tf

model_dir_define = "./adult"
train_file_name = "./adult/adult.data"
test_file_name = "./adult/adult.test"


df_train = pd.read_csv(tf.gfile.Open(train_file_name), names=tflm.COLUMNS, skipinitialspace=True, engine="python")
df_test = pd.read_csv(tf.gfile.Open(test_file_name), names=tflm.COLUMNS, skipinitialspace=True, skiprows=1, engine="python")

df_train = df_train.dropna(how='any', axis=0)
df_test = df_test.dropna(how='any', axis=0)

df_train[tflm.LABEL_COLUMN] = (df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
df_test[tflm.LABEL_COLUMN] = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

model_dir = tempfile.mkdtemp() if not model_dir_define else model_dir_define
print("model directory = %s" % model_dir)

model = tflm.build_estimator(model_dir, 'wide')
model.fit(input_fn=lambda: tflm.input_fn(df_train), steps = 200)
results = model.evaluate(input_fn=lambda: tflm.input_fn(df_test), steps = 1)

for key in sorted(results):
    print "%s: %s" % (key, results[key])









