"""
This module will takes a tweet as a string and depending on the method.

Assigns a score. (based on the average score of the 9 classifiers)

Assigns a Classification of Positive, Negative or Unsure.

I might update it later but this method will use the SGD models trained in Kaggle_Program.Train_Kaggle.py
When Num_Features=5000.

This is slightly faster, but otherwise indistinguishable,  from the accuracy of Num_Features =10000
"""

from NaturalLanguage.custom_NLTK_Utils import Pickle_Utils
import nltk
import numpy as np
from NaturalLanguage.custom_NLTK_Utils import VoteClassifier

myVoteClassifier = Pickle_Utils.unpickle_this(r"C:\Users\parke\Documents\GitHub\NaturalLanguage\NaturalLanguage\OwnPrograms\Kaggle_Program\TrainedVoteClassifier_N2500.pickle")

def get_sentiment(raw_tweet, consensus_choice=5):

    category = myVoteClassifier.classify(raw_tweet, consensus=consensus_choice)
    return category


sample_tweet = 'sad mad bad ugly hate taxes unhappy'
ans = get_sentiment(sample_tweet)

print(ans)


# you would also want to see the false positve and false negative rate.
