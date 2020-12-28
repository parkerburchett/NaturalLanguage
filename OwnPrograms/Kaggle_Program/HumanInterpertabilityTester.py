"""
This is where I am testing methods to explain certain why a prediction was reached.

# I am wrong here. Weights do not necessarily correspond to importance.
# read this https://machinelearningmastery.com/calculate-feature-importance-with-python/

# you should look at # SGDClassifier.coef. I am unsure how to get that. right now you can look at in the debugger

# you might want to use decision_function()

"""

from NaturalLanguage.custom_NLTK_Utils import Pickle_Utils
from scipy.stats.stats import pearsonr
from NaturalLanguage.custom_NLTK_Utils import VoteClassifier
import matplotlib.pyplot as plt
import numpy as np

my_voteClassifier = Pickle_Utils.unpickle_this(r'C:/Users/parke/Documents/GitHub/NaturalLanguage/NaturalLanguage/OwnPrograms/Kaggle_Program/TrainedVoteClassifier_N6000.pickle')

test_tweet = 'I am happy'

print(my_voteClassifier.get_scores(test_tweet))






# this should give you 9*8 or 72 r values.
# you could also look at the the median or the mean R values.

# each word features has 9 weight vectors associated with it.
# You could use this to see what features are the most important accross the features.
