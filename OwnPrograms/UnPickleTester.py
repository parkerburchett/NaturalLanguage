# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 14:16:55 2020

@author: parke
"""

import pickle

print("started")
loadData = open("pickled_documents.pickle", "rb")
myData = pickle.load(loadData)
loadData.close()
print(type(myData))
print("ended")