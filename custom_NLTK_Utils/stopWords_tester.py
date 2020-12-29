# this is where I am figuring out what words to exclude as stopwords.

from NaturalLanguage.custom_NLTK_Utils import Pickle_Utils
from scipy.stats.stats import pearsonr
from NaturalLanguage.custom_NLTK_Utils import VoteClassifier
import matplotlib.pyplot as plt
import numpy as np



my_voteClassifier = Pickle_Utils.unpickle_this(r'C:/Users/parke/Documents/GitHub/NaturalLanguage/NaturalLanguage/OwnPrograms/Kaggle_Program/TrainedVoteClassifier_N6000.pickle')


# first step is a list of the N most freqnet words where the weights are less then some theshhold.


word_features = my_voteClassifier.get_word_features()
weight_threshold = .5
weight_dict = my_voteClassifier.get_avg_weight_dictionary()
# the number of words that have a weight greater than some threshold.
num_non_trivial = 0
for i in range(6000):
    word = word_features[i]
    weight = weight_dict[word_features[i]]
    if(abs(weight) > weight_threshold):
        num_non_trivial +=1
        print(word, weight)

print(num_non_trivial)