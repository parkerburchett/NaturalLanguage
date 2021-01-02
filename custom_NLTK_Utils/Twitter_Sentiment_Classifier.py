"""
This module will takes a tweet as a string and depending on the method.

Assigns a score. (based on the average score of the 9 classifiers)

Assigns a Classification of Positive, Negative or Unsure.

I might update it later but this method will use the SGD models trained in Kaggle_Program.Train_Kaggle.py
When Num_Features=5000.

This is slightly faster, but otherwise indistinguishable,  from the accuracy of Num_Features =10000
"""

from NaturalLanguage.custom_NLTK_Utils import Pickle_Utils

myVoteClassifier = Pickle_Utils.unpickle_this(r"C:\Users\parke\Documents\GitHub\NaturalLanguage\NaturalLanguage\OwnPrograms\Kaggle_Program\TrainedVoteClassifier_N10000.pickle")

def get_sentiment(raw_tweet, consensus_choice=5):
    category = myVoteClassifier.classify(raw_tweet, consensus=consensus_choice)
    return category

def get_votes(raw_tweet):
    return myVoteClassifier.get_category_votes(raw_tweet)


def explain_choice(raw_tweet):
    return myVoteClassifier.explain_choice(raw_tweet)


sample_tweet = 'I hate my life, i just want to leave'

explained = explain_choice(sample_tweet)

print(explained)



# you would also want to see the false positve and false negative rate.
