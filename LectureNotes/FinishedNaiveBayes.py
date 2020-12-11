"""
https://www.youtube.com/watch?v=rISOsUaTrO4&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL&index=13


This is where we build the naive bayes algorithm to classify a review as 
negative or postiive syntiment. 

The Naive Bayse is pretty good and very cheap timecost. So it can get scaled up
a tremendous amount 

The time complexity is O(nK) (linear time)

where n is the number of features( this is the number of words)

and k is the number of labed classes ( I don't know what this means)




From wikipedia
https://en.wikipedia.org/wiki/Naive_Bayes_classifier

The Naive Bayes is a method for developing a system classifying a vector.
The vector is a very long series of words in this context. 

    This ignores corralation between values in the  vector


# posterior = prior occurances x liklhood  / current evidence


The reason why the value changes so much is that 
you are shuffling the Train, Test Data

You can see more details by looking at the variables after you run this. 



"""

import random

import nltk
from nltk.corpus import movie_reviews

# this is an ugly one liner
# documents is a list of tuples of (contents of a review, pos or negaitve)
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words =[]

for w in movie_reviews.words():
    all_words.append(w.lower())
    
all_words = nltk.FreqDist(all_words) # this is too goddam huge.

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
classifier = nltk.NaiveBayesClassifier.train(TrainingSet)

print("Accuracy of Naive Bays in Percentage:", (nltk.classify.accuracy(classifier, TestingSet)*100))

classifier.show_most_informative_features(10)
