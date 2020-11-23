# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 09:39:00 2020
https://www.youtube.com/watch?v=vlTQLb_a564&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL&index=16

Combining ALgorithms with a vote.

This is good practice because it raises accuracy and reliability.

It will also let you see the "confidence", as a number of votes from each algorithm


You can now, use this to get confidence intervals for sentimat for any type of text, 
You can do this on twitter instead

@author: parke
"""



import random, pickle, nltk

from nltk.corpus import movie_reviews


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


def find_Features(document):
    words = set(document) 
    features = {}
    for w in word_features:
        features[w] = (w in words) # this a boolean
    return features  


documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)
all_words =[]
for w in movie_reviews.words():
    all_words.append(w.lower())   
all_words = nltk.FreqDist(all_words) 
word_features = list(all_words.keys())[:3000]


feature_sets = [(find_Features(rev), category) 
                for (rev, category) in documents]

TrainingSet = feature_sets[:1900]
TestingSet = feature_sets[1900:]

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

# pass the voted classifer any number of classifer objects and it will be another 
# classifer object that classifes based on what the other classifers vote for.

voted_classifier = VoteClassifier(classifier_NaiveBayes,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  SGDClassifier_classifier,
                                  LinearSVC_classifier)

print("Accuracy of voted_classifier in Percentage :", (nltk.classify.accuracy
                                                       (voted_classifier, 
                                                        TestingSet)*100))


for i in range(100):
    print("Classification: ", voted_classifier.classify(TestingSet[i][0]), 
          "Confidence %: ", voted_classifier.confidence(TestingSet[i][0])*100)

