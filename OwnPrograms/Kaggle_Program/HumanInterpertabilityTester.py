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

def create_feature_coef_list(word_features, coef):
    """
    This method returns the word_feature, coef pair

    It shows the coeffeicnt assigned to each word_feature
    """
    pairs = []
    for i in range(len(word_features)):
        a_pair = (word_features[i],coef[0][i])
        pairs.append(a_pair)  # might be better as a dictionary. It will be better as a dictionary. Change later
    return pairs

my_voteClassifier = Pickle_Utils.unpickle_this(r'C:/Users/parke/Documents/GitHub/NaturalLanguage/NaturalLanguage/OwnPrograms/Kaggle_Program/TrainedVoteClassifier_N6000.pickle')

a_classifier = my_voteClassifier.get_classifier_list()[0]
weights1 = a_classifier.coef_[0]
a_classifier = my_voteClassifier.get_classifier_list()[1]
weights2 = a_classifier.coef_[0]
r_value = pearsonr(weights1,weights2)[0]

print(my_voteClassifier.get_relevent_words('the cow was walking down the street gibberish'))

weights_list = []



for a_classifier in my_voteClassifier.get_classifier_list():
    weights_list.append(a_classifier.coef_[0])


# this should give you 9*8 or 72 r values.
# you could also look at the the median or the mean R values.

# each word features has 9 weight vectors associated with it.
# You could use this to see what features are the most important accross the features.
