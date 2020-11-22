"""

https://www.youtube.com/watch?v=ReakZVh2Xwk&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL&index=14
Created on Sat Nov 21 17:15:22 2020

@author: parke


How to pickle an algorithm
"""

import random

import nltk
from nltk.corpus import movie_reviews
import pickle

# this is an ugly one liner
# documents is a list of tuples of (contents of a review, pos or negaitve)
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words =[]

for w in movie_reviews.words():
    all_words.append(w.lower())
    
all_words = nltk.FreqDist(all_words) # this is too goddamn huge.

word_features = list(all_words.keys())[:3000]
# only check against the first 3000 words

def find_Features(document):
    words = set(document) # a set has only unique values, this removes duplicates
    features = {}
    for w in word_features:
        features[w] = (w in words) # this a boolean
            # this says if a word in in the first 3000 words of a dataset
            # then it is that value is TRUE else it is false. 
    return features     
    

# this creates a list of tuples of (an entire review : BOOLEAN if in top 3000 words, Negative  or Positive)   
feature_sets = [(find_Features(rev), category) for (rev, category) in documents]

TrainingSet = feature_sets[:1900]
TestingSet = feature_sets[1900:]

# create a classifer object using the NaiveBayesClassifer
#classifier = nltk.NaiveBayesClassifier.train(TrainingSet)
classifier_f = open("naivebayes.pickle", "rb") # get the picked naive bayse classifer

classifier = pickle.load(classifier_f)
classifier_f.close()

print("Accuracy of Naive Bays in Percentage:", (nltk.classify.accuracy(classifier, TestingSet)*100))

classifier.show_most_informative_features(10)

#save_classifer =  open("naivebayes.pickle", "wb") "wb" means write in bytes
#pickle.dump(classifier, save_classifer)
#save_classifer.close()



