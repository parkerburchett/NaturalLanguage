# -*- coding: utf-8 -*-
"""
I needed to store the pickled classifiers somewhere.

I am creating a list of Classifer objects and 
am pickling that to save time using the programs in the future. 


https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
This is a beautiful way of looking at it. 

    
"""

import random, pickle, nltk
import datetime
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier 
from nltk.classify import ClassifierI
from statistics import mode
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.corpus import stopwords


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


def create_Labeled_Data(PositiveExamples, NegExamples):
    documents = [] # document is a tuple of (review, classifcation)
    for r in PositiveExamples.split('\n'):
        documents.append((r,"pos"))
    for r in NegExamples.split('\n'):
        documents.append((r,"neg"))
        
    all_words = []
    short_pos_words = nltk.word_tokenize(shortPos)
    short_neg_words = nltk.word_tokenize(shortNeg)
    for w in short_pos_words:
        all_words.append(w.lower()) 
    for w in short_neg_words:
        all_words.append(w.lower())
        
    stop_words = set(stopwords.words('english')) # I added this to remove all the stop words
    print(len(all_words))
    all_words = [w for w in all_words if (not w in stop_words)] 
    print("after removing stopwords")
    print(len(all_words))
    all_words = nltk.FreqDist(all_words)
    word_features = list(all_words.keys())[:3000]
    
    feature_sets = [(find_Features(rev, word_features), category) 
                    for (rev, category) in documents]
    return feature_sets
        
    # only works with SklearnClassifier
def createClassiferList(*untrainedClassifier, TrainingSet):
    classifiers = []
    for c in untrainedClassifier:
        cur  = SklearnClassifier(c())
        cur.train(TrainingSet)
        classifiers.append(cur)


start = datetime.datetime.now()

print('you have started')
untrainedClassifier = [SGDClassifier()]

shortPos = open("short_reviews/shortPositive.txt","r").read()
shortNeg = open("short_reviews/shortNegative.txt","r").read()

LabeledReviews = create_Labeled_Data(shortPos, shortNeg)
random.shuffle(LabeledReviews)

cleanedReviews = open("cleanedReviews.pickle", "wb")
pickle.dump(LabeledReviews, cleanedReviews)
cleanedReviews.close()


TrainingData = LabeledReviews[:9000]
TestingData = LabeledReviews[9000:]


NBClassifer = nltk.NaiveBayesClassifier.train(TrainingData)
NBClassifer.show_most_informative_features(10)

print(nltk.classify.accuracy(NBClassifer, TestingData)*100)







print("the program took this long: ")
end = datetime.datetime.now()
print(end-start)


