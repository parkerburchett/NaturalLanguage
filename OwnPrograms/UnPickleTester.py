# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 14:16:55 2020

@author: parke
"""

import pickle


loadData = open("pickled_thingToPickle.pickle", "rb")
myData = pickle.load(loadData)

print(type(myData))