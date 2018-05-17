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

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.datasets import load_boston



# building dataset 

boston = load_boston()



# creating 2 list for savin results

training_accuracy = []
test_accuracy = []



# splitin dataset 

X_train, X_test, y_train, y_test = train_test_split(
	boston.data, boston.target, random_state=42)



# Linear Regression

lr_settings = (True, False)

for norm in lr_settings : 

	lr = LinearRegression(fit_intercept = True, normalize = norm)\
		.fit(X_train, y_train)

	# record training set accuracy
	train_score = lr.score(X_train, y_train)
	training_accuracy.append([train_score, "LinearRegression", norm])

	# record generalization accuracy
	test_score = lr.score(X_test, y_test)
	test_accuracy.append([test_score,  "LinearRegression", norm])

	# printing results
	print("for {}, with normalize as {}".format("LinearRegression", norm)) 
	print("train score: {:.2f}, test score: {:.2f}\n"
			.format(train_score, test_score))


# Ridge

solver_settings = ( 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 
					'sag', 'saga')

alpha_settings = np.logspace(-5,5, 11) 

for norm in lr_settings : 

	for solv in solver_settings : 

		for alph in alpha_settings : 

			# create and fit the model
			rdg = Ridge(alpha = alph, fit_intercept=True, normalize=norm, 
					solver=solv).fit(X_train, y_train)

			# record training set accuracy
			train_score = rdg.score(X_train, y_train)
			training_accuracy.append([train_score, "Ridge", norm, solv, alph])

			# record generalization accuracy
			test_score = rdg.score(X_test, y_test)
			test_accuracy.append([test_score,  "Ridge", norm, solv, alph])

			# printing results
			print("for {}, with norm {}, solv {}, alph {}"
					.format("Ridge", norm, solv, alph)) 
			print("train score: {:.2f}, test score: {:.2f}\n"
					.format(train_score, test_score))



training_accuracy.sort(reverse=True)
test_accuracy.sort(reverse=True)
print(training_accuracy[:4])
print(test_accuracy[:4])

