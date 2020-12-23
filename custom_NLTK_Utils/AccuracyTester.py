# -*- coding: utf-8 -*-
"""
When you run this without any of the code commented out it will generate 
.pickle files for all of the classifiers and all of the intermediate steps.

Note: you don't shuffle the feature sets each run so they will always be the same


It takes about 20 minutes to train all the algos, 15 minutes of that is nuSCV

https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
This is a beautiful way of looking at it. 

    
"""
import NaturalLanguage.custom_NLTK_Utils.dataLabeling as DL
import NaturalLanguage.custom_NLTK_Utils.Pickle_Utils
import NaturalLanguage.custom_NLTK_Utils.VoteClassifier
import random, pickle, nltk
import datetime
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier 
from nltk.classify import ClassifierI
from statistics import mode
from sklearn.svm import SVC, LinearSVC, NuSVC


start = datetime.datetime.now()
print('This is the buggy version you have started')

shortPos = open("C:/Users/parke/Documents/GitHub/NaturalLanguage/NaturalLanguage/custom_NLTK_Utils/short_reviews/shortPositive.txt","r").read()
shortNeg = open("C:/Users/parke/Documents/GitHub/NaturalLanguage/NaturalLanguage/custom_NLTK_Utils/short_reviews/shortNegative.txt","r").read()


feature_sets =  DL.create_feature_sets(shortPos, shortNeg)
N = int(len(feature_sets)*.9)
TrainingData = feature_sets[:N]
TestingSet = feature_sets[N:]


print("Buggy now training")
TrainedClassifierList = []
NBClassifer = nltk.NaiveBayesClassifier.train(TrainingData)
TrainedClassifierList.append(NBClassifer)
print("Buggy Trained")
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

voted_classifier = NaturalLanguage.custom_NLTK_Utils.VoteClassifier.VoteClassifier(TrainedClassifierList[0],
                                                      TrainedClassifierList[1],
                                                      TrainedClassifierList[2],
                                                      TrainedClassifierList[3],
                                                      TrainedClassifierList[4])

print("BUGGY Accuracy of Naive Bayes:", (nltk.classify.accuracy(TrainedClassifierList[0], 
                                                          TestingSet)*100))

print("Accuracy of voted_classifier in Percentage :", (nltk.classify.accuracy
                                                        (voted_classifier, 
                                                        TestingSet)*100))

print("BUGGY the program took this long: ")
print(datetime.datetime.now() -start)


