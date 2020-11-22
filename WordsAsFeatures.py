"""
https://www.youtube.com/watch?v=-vVskDsHcVc&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL&index=12


Words as Features for Learning




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
    


#print((find_Features(movie_reviews.words('neg/cv000_29416.txt'))))

# this creates a list of tuples of ()   
feature_sets = [(find_Features(rev), category) for (rev, category) in documents]
# now it will be the words with true or false
# i don't reall understand this part. He says that this will make the data to test agains
# this will make a set that looks like this in

# "WORD": does the word occur in the top 3000 most frequent words?

#print(feature_sets[0][0])


