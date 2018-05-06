#!/usr/bin/env python3
# -*- coding: utf-8 -*-



################################################################################
#       First application
################################################################################



# import

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
# %matplotlib inline

import seaborn as sns
sns.set()
 
from sklearn.model_selection import *
from sklearn.neighbors import KNeighborsClassifier



# load_iris function:

from sklearn.datasets import load_iris
iris = load_iris()



# About iris dataset

# show iris data
print(iris.keys())

# The value with key target_names is an array of strings, containing the species of
# flower that we want to predict:
print(iris['target_names'])


# The data itself is contained in the target : 
print(type(iris['data']))


print(iris['data'].shape)


# Here are the feature values for the first five samples:
print(iris['data'][:5])


# The target array contains the species
type(iris['target'])
type(iris['target'][:20])



# Creation training and testing data set

X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'],
	random_state=0)

print(X_train.shape)
print(X_test.shape)



# Short exploration

fig, ax = plt.subplots(3, 3, figsize=(15, 15))	
plt.suptitle("iris_pairplot")

for i in range(3):
	for j in range(3):
		ax[i, j].scatter(X_train[:, j], X_train[:, i + 1], c=y_train, s=60)
		ax[i, j].set_xticks(())
		ax[i, j].set_yticks(())
		if i == 2:
			ax[i, j].set_xlabel(iris['feature_names'][j])
		if j == 0:
			ax[i, j].set_ylabel(iris['feature_names'][i + 1])
		if j > i:
			ax[i, j].set_visible(False)
plt.show()




# Building your first model:

# creating and fiting KNN
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)



# Making predictions

# try a random prediction
X_new = np.array([[5, 2.9, 1, 0.2]])
print(X_new.shape)

prediction = knn.predict(X_new)

print(prediction) 
print(iris['target_names'][prediction])



# Evaluating the model

y_pred = knn.predict(X_test)
result = np.mean(y_pred == y_test)
print(result)

result = knn.score(X_test, y_test)
print(result)
