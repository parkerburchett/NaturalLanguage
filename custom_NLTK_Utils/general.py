import random, pickle, nltk
from nltk.classify import ClassifierI
from statistics import mode
from nltk.corpus import stopwords
# https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string
                
"""
These are both methods that I have written and methods from the sentdex NLTK lecture series

Source: https://www.youtube.com/watch?v=eObouMO2qSE&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL&index=19


"""

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


def customPickle(thingToPickle, objectName): 
    """
        Pass this an Object and it will create a pickled instance of it
        it will be save in this format "pickled_"+ name +".pickle"
        
        eg You want to pickle a list called myList
        
        this will create a file called 
        pickled_myList.pickle
    """

    objectName = "pickled_"+ objectName +".pickle"
    outLocation = open(objectName, "wb")
    pickle.dump(thingToPickle, outLocation)
    outLocation.close()
    
    


def pickle_Intermediate_Steps(PositiveExamples, NegativeExamples):
    """
    This is very similar to the create_feature_sets() but instead of returning
    a feature_set it pickles the following:
        documents
        all_words
        word_features
        feature_sets
        
    """
    documents = [] # document is a tuple of (review, classifcation)
    for r in PositiveExamples.split('\n'):
        documents.append((r,"pos"))
    for r in NegativeExamples.split('\n'):
        documents.append((r,"neg"))
        
    customPickle(documents,"documents")
    
    all_words = []
    short_pos_words = nltk.word_tokenize(PositiveExamples)
    short_neg_words = nltk.word_tokenize(NegativeExamples)
    
    for w in short_pos_words:
        all_words.append(w.lower()) 
    for w in short_neg_words:
        all_words.append(w.lower())
    
 
    stop_words = set(stopwords.words('english')) # I added this to remove all the stop words
    all_words = [w for w in all_words if (not w in stop_words)] 
    all_words = nltk.FreqDist(all_words)
    customPickle(all_words,"all_words")
    
    word_features = list(all_words.keys())[:5000] # 5000 is an arbritary choice
    
    customPickle(word_features,"word_features")
    
    feature_sets = [(find_Features(rev, word_features), category) 
                    for (rev, category) in documents]
    
    random.shuffle(feature_sets)
    customPickle(feature_sets,"feature_sets")

    
    
def create_feature_sets(PositiveExamples, NegativeExamples):
    documents = [] # document is a tuple of (review, classifcation)
    for r in PositiveExamples.split('\n'):
        documents.append((r,"pos"))
    for r in NegativeExamples.split('\n'):
        documents.append((r,"neg"))
    all_words = []
    
    short_pos_words = nltk.word_tokenize(PositiveExamples)
    short_neg_words = nltk.word_tokenize(NegativeExamples)
    
    for w in short_pos_words:
        all_words.append(w.lower()) 
    for w in short_neg_words:
        all_words.append(w.lower())

    stop_words = set(stopwords.words('english')) # I added this to remove all the stop words
    all_words = [w for w in all_words if (not w in stop_words)] 
    all_words = nltk.FreqDist(all_words)
    word_features = list(all_words.keys())[:3000]
    
    feature_sets = [(find_Features(rev, word_features), category) 
                    for (rev, category) in documents]
    
    return random.shuffle(feature_sets)