# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:35:46 2019

@author: alexa
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore', categorical_features=[0])
X = [['Male', 1], ['Female', 3], ['Female', 2]]