# -*- coding: utf-8 -*-
"""
When you run this without any of the code commented out it will generate 
.pickle files for all of the classifiers and all of the intermediate steps.

Note: you don't shuffle the feature sets each run so they will always be the same


It takes about 20 minutes to train all the algos, 15 minutes of that is nuSCV

https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
This is a beautiful way of looking at it. 

    
"""

import NaturalLanguage.custom_NLTK_Utils.Inital_Pickle
import NaturalLanguage.custom_NLTK_Utils.VoteClassifier


import random, pickle, nltk
import datetime
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier 
from nltk.classify import ClassifierI
from statistics import mode
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.corpus import stopwords

start = datetime.datetime.now()
print('you have started')

# shortPos = open("short_reviews/shortPositive.txt","r").read()
# shortNeg = open("short_reviews/shortNegative.txt","r").read()

# general.pickle_Intermediate_Steps(shortPos,shortNeg)



loadFS = open("pickled_feature_sets.pickle", "rb")
feature_sets = pickle.load(loadFS)
loadFS.close()


TrainingData = feature_sets[:9000]
TestingData = feature_sets[9000:]



TrainedClassifierList = []
NBClassifer = nltk.NaiveBayesClassifier.train(TrainingData)
TrainedClassifierList.append(NBClassifer)
print(datetime.datetime.now() -start)

c = SklearnClassifier(SGDClassifier())
c.train(TrainingData)
TrainedClassifierList.append(c)
print(datetime.datetime.now() -start)

c = SklearnClassifier(BernoulliNB())
c.train(TrainingData)
TrainedClassifierList.append(c)
print(datetime.datetime.now() -start)

c = SklearnClassifier(LinearSVC())
c.train(TrainingData)
TrainedClassifierList.append(c)
print(datetime.datetime.now() -start)

c = SklearnClassifier(LogisticRegression())
c.train(TrainingData)
TrainedClassifierList.append(c)
print(datetime.datetime.now() -start)

cs.Inital_Pickle.customPickle(TrainedClassifierList, "TrainedClassifierList")


print("the program took this long: ")
end = datetime.datetime.now()
print(end-start)


