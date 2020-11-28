# -*- coding: utf-8 -*-
"""
Better Trainging Data

Source:
https://www.youtube.com/watch?v=UF-RyxOAHQw&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL&index=18


Before you finish you finish this lecture you need to pickle all of the following:
    
    documents, all_words, feature_sets,
    
    ALl of the classifiers,
    
    importantn to note the program stopped started running very slowly at NuSVC_classifier.
    
    You should get rid of that infavor of another binary classifier.
    
    
    Look at the notes on pickling to see the accuracy of that. 
    





    
"""

import random, pickle, nltk
from nltk.corpus import movie_reviews
import datetime
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier 
from nltk.classify import ClassifierI
from statistics import mode;
from sklearn.svm import SVC, LinearSVC, NuSVC

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        # the init method will always run in a class
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf =  choice_votes / len(votes)
        return conf

def find_Features(document, word_features):
    words = nltk.word_tokenize(document) 
    features = {}
    for w in word_features:
        features[w] = (w in words) # this a boolean
    return features  



start = datetime.datetime.now()

print('you have started')


shortPos = open("short_reviews/shortPositive.txt","r").read()

shortNeg = open("short_reviews/shortNegative.txt","r").read()

documents = [] # document is a tuple of (review, classifcation)
# all words is a list of every word. 
# documents is a list of review objects
for r in shortPos.split('\n'):
    documents.append((r,"pos"))
    
for r in shortNeg.split('\n'):
    documents.append((r,"neg"))
    
all_words = []

short_pos_words = nltk.word_tokenize(shortPos)
short_neg_words = nltk.word_tokenize(shortNeg)

for w in short_pos_words:
    all_words.append(w.lower())
    
for w in short_neg_words:
    all_words.append(w.lower())
    
all_words = nltk.FreqDist(all_words) 
word_features = list(all_words.keys())[:3000]
feature_sets = [(find_Features(rev, word_features), category) for (rev, category) in documents]

random.shuffle(feature_sets)

TrainingSet = feature_sets[:10000]
TestingSet = feature_sets[10000:]


classifier_NaiveBayes = nltk.NaiveBayesClassifier.train(TrainingSet)
print("Original Accuracy of Naive Bays in Percentage:", 
      (nltk.classify.accuracy(classifier_NaiveBayes, TestingSet)*100))

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(TrainingSet)
print("Accuracy of MNB_classifier in Percentage:", 
      (nltk.classify.accuracy(MNB_classifier, TestingSet)*100))

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(TrainingSet)
print("Accuracy of BernoulliNB_classifier in Percentage:", 
      (nltk.classify.accuracy(BernoulliNB_classifier, TestingSet)*100))

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(TrainingSet)
print("Accuracy of SGDClassifier_classifier in Percentage:", 
      (nltk.classify.accuracy(SGDClassifier_classifier, TestingSet)*100))


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(TrainingSet)
print("Accuracy of LinearSVC_classifier in Percentage:", 
      (nltk.classify.accuracy(LinearSVC_classifier, TestingSet)*100))

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(TrainingSet)
print("Accuracy of NuSVC_classifier in Percentage:", 
      (nltk.classify.accuracy(NuSVC_classifier, TestingSet)*100))



voted_classifier = VoteClassifier(classifier_NaiveBayes,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  SGDClassifier_classifier,
                                  LinearSVC_classifier)

print("Accuracy of voted_classifier in Percentage :", (nltk.classify.accuracy
                                                       (voted_classifier, 
                                                        TestingSet)*100))




print("the program took this long: ")
end = datetime.datetime.now()
print(end-start)

