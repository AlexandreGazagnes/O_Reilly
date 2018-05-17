#!/usr/bin/env python3
# -*- coding: utf-8 -*-



################################################################################
#       KNN
################################################################################



# import

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
# %matplotlib inline

import seaborn as sns
sns.set()

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.datasets import load_breast_cancer



# building dataset 

cancer = load_breast_cancer()



# splitin dataset 

X_train, X_test, y_train, y_test = train_test_split(
	cancer.data, cancer.target, stratify=cancer.target, random_state=66)



# creating 2 list for savin results

training_accuracy = []
test_accuracy = []



# try n_neighbors from 1 to 10.

neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:

	# build the model
	clf = KNeighborsClassifier(n_neighbors=n_neighbors)
	clf.fit(X_train, y_train)

	# record training set accuracy
	training_accuracy.append(clf.score(X_train, y_train))

	# record generalization accuracy
	test_accuracy.append(clf.score(X_test, y_test))



# ploting results

plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.legend()
plt.show()


# two important parameters to the KNeighbors classifier: 
#  - number of neighbors 
#  - how you measure distance between data points. 

# strengths of nearest neighbors : 
# 	the model is very easy to understand,
# 	often gives reasonable performance without a lot of adjustments. 
# 	good baseline method
# 	fast and easy 

# weakness : 
# 	preprocessin for numerical data
# 	not good for large (sample and features) data (slow and expensive)



