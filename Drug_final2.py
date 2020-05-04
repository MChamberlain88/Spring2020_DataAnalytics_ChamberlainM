
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:15:02 2020

@author: chambm6
"""
#Import necessary packages
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
import sklearn
assert sklearn.__version__ >= "0.20"
import tensorflow as tf
assert tf.__version__ >= "2.0"
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.metrics
import sklearn.neural_network
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
#Reading in Drug Consumption data
df = pd.read_csv("C:/Users/chambm6/Desktop/Data_Analytics/drug_consumption.data.csv")
#Removing participants who answered to taking a fictional drug called semer
fake = df[df['semer']!='CL0']
df = df.drop([df.index[727], df.index[817], df.index[1516], df.index[1533], df.index[1698], df.index[1769], df.index[1806], df.index[1823]])

#List of attribute names
feature_col_names = ['age', 'gender', 'education', 'country', 'ethnicity', 'nscore',
       'escore', 'oscore', 'ascore', 'cscore', 'impulsive', 'ss']
#List of drugs
columns = ['alcohol','amphet', 'amyl', 'benzos', 'caff', 'cannabis', 'choc', 'coke', 'crack',
           'ecstasy', 'heroin', 'ketamine', 'legalh', 'LSD', 'meth', 'mushrooms','nicotine', 'semer', 'VSA']
#Transforming taret variable values to integers corresponding to class names
for column in columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
#Drug name we are looking to predict
predicted_class_names = ['cannabis']
#Creating training attributes and target variable
X = df[feature_col_names].values
y = df[predicted_class_names].values
#Splitting data into training and testing sets
xtrain, X_test, ytrain, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

#Building KNN Classifier
clf = KNeighborsClassifier(n_neighbors = 60, weights = 'distance')
#Fitting model on training data
clf.fit(xtrain,ytrain.ravel())
#Predicting class labels
pred = clf.predict(X_test)
#Calculating testing accuracy
accu = metrics.accuracy_score(y_test, pred)
print("Accuracy: %.2f%%" % (accu * 100.0))

import seaborn as sns
sns.set_style("darkgrid")
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

#Calculating and graphing feature importance
results = permutation_importance(clf, xtrain, ytrain, scoring='accuracy', n_repeats = 5)
# get importance
importance = results.importances_mean
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance

plt.bar([x for x in range(len(importance))], importance, color = 'steelblue')
plt.xticks(np.arange(12),labels= feature_col_names, rotation = 90)
plt.show()
