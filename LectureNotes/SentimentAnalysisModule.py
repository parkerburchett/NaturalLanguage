
"""
Sentiment Analysis Module.

In this lecture we take what we already have and link it up with the Twitter API
So that we can get live sentiment analysis from twitter by topics.

In the lecture he only looks at Adjectives. using nltk in the word tokenize section

I am going to ignore that and just use the algos I have now



Using the lookatAccuracy Method you can see when it is right about the classification

important to note its accuracy is only 76% when it says it is 100% confident


Source:
    https://www.youtube.com/watch?v=eObouMO2qSE&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL&index=19
"""


from NaturalLanguage.custom_NLTK_Utils import Inital_Pickle
from NaturalLanguage.custom_NLTK_Utils import dataLabeling as DL
from NaturalLanguage.custom_NLTK_Utils import VoteClassifier
import random, pickle, nltk

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


voted_classifier = VoteClassifier.VoteClassifier(classifiers[0],classifiers[1],
                                          classifiers[2],classifiers[3],
                                          classifiers[4])


def sentiment(text):
    """
    

    Parameters
    ----------
    text : String
        A plain string test that you are attempting to find the sentiment of

    Returns
    -------
    The voted_classifers guess of if the text is positive or negative,
    and how confident it is
    
    example
       sentiment("The movie was bad and awful")
           -> neg 1.0
           
    This can be interperated as the algo is 100% confident that sentiment is negative
    
    """
    feats = DL.find_Features(text, word_features)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)
