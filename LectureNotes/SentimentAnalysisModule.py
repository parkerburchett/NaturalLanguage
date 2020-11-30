
"""
Sentiment Analysis Module.

In this lecture we take what we already have and link it up with the Twitter API
So that we can get live sentiment analysis from twitter by topics.

In the lecture he only looks at Adjectives. using nltk in the word tokenize section

I am going to ignore that and just use the algos I have now




Source:
    https://www.youtube.com/watch?v=eObouMO2qSE&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL&index=19
"""


from NaturalLanguage.custom_NLTK_Utils import general
import random, pickle, nltk
import datetime
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier 
from nltk.classify import ClassifierI
from statistics import mode;
from sklearn.svm import SVC, LinearSVC, NuSVC


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


voted_classifier = general.VoteClassifier(classifiers)


def sentiment(text):
    feats = general.find_Features(documents, word_features)
    print(voted_classifier.classify(feats))
    return voted_classifier.classify(feats)



sentiment("I thought the movie was bad and boring")