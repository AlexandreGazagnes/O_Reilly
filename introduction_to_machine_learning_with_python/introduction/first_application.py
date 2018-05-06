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

from sklearn.preprocessing import * 
from sklearn.model_selection import *
from sklearn.linear_model import * 
from sklearn.metrics import r2_score
from sklearn.ensemble import *



# load_iris function:
from sklearn.datasets import load_iris
iris = load_iris()



# show iris data
print(iris.keys())



# The value with key target_names is an array of strings, containing the species of
# flower that we want to predict:
print(iris['target_names'])


# The data itself is contained in the target : 
print(type(iris['data']))