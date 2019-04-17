import numpy as np
import pandas as pd
import tensorflow as tf
from subprocess import check_output
from sklearn.model_selection import train_test_split
# print(check_output(["ls", "../DATAMINING"]).decode("utf8"))

def input_fn(df,labels):
    feature_cols = {k:tf.constant(df[k].values,shape = [df[k].size,1]) for k in columns}
    label = tf.constant(labels.values, shape = [labels.size,1])
    return feature_cols,label

CA = pd.read_csv("../DATAMINING/temp.csv")
# print(CA)

# print(CA.shape)
# print(CA.dtypes)


CA["Cat1"] = CA["Cat1"].map({"a" : 0 , "b" : 1})
CA["Cat2"] = CA["Cat2"].map({"u" : 0 , "y" : 1 , "l" : 2 , "t" : 3})
CA["Cat3"] = CA["Cat3"].map({"g" : 0 , "p" : 1 , "gg" : 2})
CA["Cat4"] = CA["Cat4"].map({"c" : 0 , "d" : 1 , "cc" : 2 , "i" : 3 , "j" : 4 , "k" : 5 , "m" : 6 , "r" : 7 , "q" : 8 , "w" : 9 , "x" : 10 , "e" : 11 , "aa" : 12 , "ff" : 13})
CA["Cat5"] = CA["Cat5"].map({"v" : 0 , "h" : 1 , "bb" : 2 , "j" : 3 , "n" : 4 , "z" : 5 , "dd" : 6 , "ff" : 7, "o" : 8})
CA["Cat6"] = CA["Cat6"].map({"f" : 0 , "t" : 1})
CA["Cat7"] = CA["Cat7"].map({"f" : 0 , "t" : 1})
CA["Cat8"] = CA["Cat8"].map({"f" : 0 , "t" : 1})
CA["Cat9"] = CA["Cat9"].map({"g" : 0 , "p" : 1 , "s" : 2})
# CA = pd.get_dummies(CA, columns=CA[0])
# print(CA)
CA["Class"] = CA["Class"].map({"+" : 0 , "-" : 1})
CA.iloc[:,0:15] = CA.iloc[:,0:15].astype(np.float64)
# print(CA.dtypes)

X_train, X_test, y_train, y_test = train_test_split(CA.iloc[:,0:15], CA["Class"], test_size=0.2, random_state=50)

# print(X_train.shape)
# print(X_test.shape)

columns = CA.columns[0:15]
# print(columns)
# print(X_test)
# print(X_train)
# print(y_test)
# print(y_train)

feature_columns = [tf.contrib.layers.real_valued_column(k) for k in columns]

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[10],n_classes = 2 )

classifier.fit(input_fn=lambda: input_fn(X_train,y_train),steps = 2000)

ev = classifier.evaluate(input_fn=lambda: input_fn(X_test,y_test),steps=1)

print(ev)