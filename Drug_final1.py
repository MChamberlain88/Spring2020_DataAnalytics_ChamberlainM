# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 14:28:04 2020

@author: chambm6
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:49:57 2020

@author: chambm6
"""

import sys
assert sys.version_info >= (3, 5)
import sklearn
assert sklearn.__version__ >= "0.20"
import tensorflow as tf
assert tf.__version__ >= "2.0"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.metrics
import sklearn.neural_network
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras import optimizers
#Read in Drug Consumption Data
df = pd.read_csv("C:/Users/chambm6/Desktop/Data_Analytics/drug_consumption.data.csv")
#Filtering out participants who answered yes to using fictional drug semer
fake = df[df['semer']!='CL0']
df = df.drop([df.index[727], df.index[817], df.index[1516], df.index[1533], df.index[1698], df.index[1769], df.index[1806], df.index[1823]])

#List of features
feature_col_names = ['age','gender', 'education', 'country','ethnicity','nscore',
       'escore', 'oscore', 'ascore', 'cscore', 'impulsive', 'ss']

#List of drug types
columns = ['alcohol','amphet', 'amyl', 'benzos', 'caff', 'cannabis', 'choc', 'coke', 'crack',
           'ecstasy', 'heroin', 'ketamine', 'legalh', 'LSD', 'meth', 'mushrooms','nicotine', 'semer', 'VSA']

#Transforming target variable class names to integers
for column in columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])

#Drug we are testing for classification, this changes for each classification run
predicted_class_names = ['cannabis']
#Creating atribute and target variable columns
X = df[feature_col_names]
y = df[predicted_class_names]
#Binarizing the target variable
dummy_y = np_utils.to_categorical(y)
#Splitting the data into a training and testing set
X_train_full, X_test, y_train_full, y_test = train_test_split(X, dummy_y, test_size=0.30, random_state=42)


#Furth splitting the training data into a training and validation set used to validate the model
xtrain, xvalid, ytrain, yvalid = train_test_split(X_train_full, y_train_full, random_state=42)

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

#Building neural network
model = keras.models.Sequential()
model.add(Dense(output_dim=30, init='uniform', activation='tanh', input_dim=12))
model.add(Dense(output_dim=20, init='uniform', activation='tanh'))
model.add(Dense(output_dim=12, init='uniform', activation='tanh'))
model.add(Dense(output_dim=7, init='uniform', activation='sigmoid'))
model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
#Fitting the model on the trainig data
keras_reg = model.fit(xtrain, ytrain, batch_size = 128, epochs = 200, verbose = 1, validation_data=(xvalid, yvalid))
#Predicting class labels
y_pred = model.predict(X_test)
for i in range(len(y_pred)):
    maxs = max(y_pred[i])
    y_pred[i] = (y_pred[i]==maxs)
#Evaluating the model
score, acc = model.evaluate(X_test, y_test, batch_size = 100, verbose = 1)
print('Test accuracy:', acc)

from sklearn.inspection import permutation_importance

#Calculating feature importance
results = permutation_importance(model, xtrain, ytrain, scoring='neg_mean_squared_error')
#get importance
importance = results.importances_mean
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance, color = 'steelblue')
plt.xticks(np.arange(12),labels= feature_col_names, rotation = 90)
plt.show()


