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

from sklearn.preprocessing import * 
from sklearn.model_selection import *
from sklearn.linear_model import * 
from sklearn.metrics import r2_score
from sklearn.ensemble import *
from sklearn.neighbors import KNeighborsClassifier
