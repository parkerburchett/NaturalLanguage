# -*- coding: utf-8 -*-
"""
I needed to store the pickled classifiers somewhere.

I am creating a list of Classifer objects and 
am pickling that to save time using the programs in the future. 


https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
This is a beautiful way of looking at it. 

    
"""

from NaturalLanguage.custom_NLTK_Utils import general

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

shortPos = open("short_reviews/shortPositive.txt","r").read()
shortNeg = open("short_reviews/shortNegative.txt","r").read()

general.pickle_Intermediate_Steps(shortPos,shortNeg)












# TrainedClassifierList = []

# NBClassifer = nltk.NaiveBayesClassifier.train(TrainingData)
# TrainedClassifierList.append(NBClassifer)
# print(datetime.datetime.now() -start)

# c = SklearnClassifier(NuSVC())
# c.train(TrainingData)
# TrainedClassifierList.append(c)
# print(datetime.datetime.now() -start)

# c = SklearnClassifier(GaussianNB())
# c.train(TrainingData)
# TrainedClassifierList.append(c)
# print(datetime.datetime.now() -start)

# c = SklearnClassifier(SGDClassifier())
# c.train(TrainingData)
# TrainedClassifierList.append(c)
# print(datetime.datetime.now() -start)

# c = SklearnClassifier(BernoulliNB())
# c.train(TrainingData)
# TrainedClassifierList.append(c)
# print(datetime.datetime.now() -start)

# c = SklearnClassifier(LinearSVC())
# c.train(TrainingData)
# TrainedClassifierList.append(c)
# print(datetime.datetime.now() -start)

# c = SklearnClassifier(LogisticRegression())
# c.train(TrainingData)
# TrainedClassifierList.append(c)
# print(datetime.datetime.now() -start)



# TrainedClassifers = open("trainedAlgos.pickle", "wb")
# pickle.dump(TrainedClassifierList,TrainedClassifers);
# TrainedClassifers.close();


# print(datetime.datetime.now() -start)



print("the program took this long: ")
end = datetime.datetime.now()
print(end-start)


