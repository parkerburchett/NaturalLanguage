"""
https://www.youtube.com/watch?v=zi16nl82AMA

This will see if a text has a positive or negative connotation. 

You can apply this to any categories as long as they are tagged and you only
have two categories



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


# the data you train on is different than the data you test on


# merge all of the words from the negative reivews into one group, 
# merge all of the words from the negaive reiews into another group.
# get the frequency of each word there


# for a new review, see if it has more negative words or positive words
# this is naive bayes algorithm



all_words =[]
for w in movie_reviews.words():
    all_words.append(w.lower())
    # all things are lowercase

# you would need to remove all the meaningless words with a stopwords set. 

all_words = nltk.FreqDist(all_words) # i think this makes a se
#print(all_words.most_common(15))

myword = "trump"
print('this is the number of times ', myword, " appears in the dataset is")
print(all_words[myword])
