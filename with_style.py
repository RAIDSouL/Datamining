from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import os

def input_fn(df,labels):
    feature_cols = {k:tf.constant(df[k].values,shape = [df[k].size,1]) for k in columns}
    label = tf.constant(labels.values, shape = [labels.size,1])
    return feature_cols,label

header = ['cat1','num1','num2',
        'cat2','cat3','cat4',
        'cat5','num3','cat6',
        'cat7','num4','cat8',
        'cat9','num5','num6',
        'class']
df = pd.read_csv("../DATAMINING/crx.dataFixed.csv" , names=header)
df['class'].replace('-',0,inplace=True)
df['class'].replace('+',1,inplace=True)

df = pd.get_dummies(df, columns=header[0:1])
df = pd.get_dummies(df, columns=header[3:7])
df = pd.get_dummies(df, columns=header[8:10])
df = pd.get_dummies(df, columns=header[11:13])

df = df.reindex(columns=(['class'] + list([a for a in df.columns if a != 'class']) ))

# print(df.shape[1])
x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,1:47], df["class"], test_size=0.1, random_state=50)

columns = df.columns[1:47]
# print(columns)

feature_columns = [tf.contrib.layers.real_valued_column(k) for k in columns]
# tf.contrib.estimator.binary_classification_head
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[10,20,10],n_classes = 2
    # ,optimizer=tf.train.ProximalAdagradOptimizer(
    #         learning_rate=0.5,
    #         # l1_regularization_strength=0.001
    #     )
        )

classifier.fit(input_fn=lambda: input_fn(x_train,y_train),steps = 2000)

ev = classifier.evaluate(input_fn=lambda: input_fn(x_test,y_test),steps=1)

print(format(ev))
# print(y_train)