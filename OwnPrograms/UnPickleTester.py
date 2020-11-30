# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 14:16:55 2020

@author: parke
"""

import pickle
import nltk
from NaturalLanguage.custom_NLTK_Utils import general

print("started")

classifierList = open("C:/Users/parke/Documents/GitHub/NaturalLanguage/NaturalLanguage/OwnPrograms/pickled_TrainedClassifierList.pickle", "rb")
classifiers = pickle.load(classifierList)
classifierList.close()

testingSetIn = open("C:/Users/parke/Documents/GitHub/NaturalLanguage/NaturalLanguage/OwnPrograms/pickled_TestingData.pickle", "rb")
TestingSet = pickle.load(testingSetIn)
testingSetIn.close()

word_featuresIN = open("C:/Users/parke/Documents/GitHub/NaturalLanguage/NaturalLanguage/OwnPrograms/pickled_word_features.pickle", "rb")
word_features = pickle.load(word_featuresIN)
word_featuresIN.close()

documentsIN = open("C:/Users/parke/Documents/GitHub/NaturalLanguage/NaturalLanguage/OwnPrograms/pickled_documents.pickle","rb")
documents = pickle.load(documentsIN)
documentsIN.close()


voted_classifier = general.VoteClassifier(classifiers[0],classifiers[1],
                                          classifiers[2],classifiers[3],
                                          classifiers[4])



print("Accuracy of voted_classifier in Percentage :", (nltk.classify.accuracy
                                                       (voted_classifier, 
                                                        TestingSet)*100))

print("ended")