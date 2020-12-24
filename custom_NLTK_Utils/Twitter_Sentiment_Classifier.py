"""
This module will takes a tweet as a string and depending on the method.

Assigns a score. (based on the average score of the 9 classifiers)

Assigns a Classification of Positive, Negative or Unsure.

I might update it later but this method will use the SGD models trained in Kaggle_Program.Train_Kaggle.py
When Num_Features=5000.

This is slightly faster, but otherwise indistinguishable  Num_Features =10000
"""

from NaturalLanguage.custom_NLTK_Utils import Pickle_Utils
from NaturalLanguage.custom_NLTK_Utils import VoteClassifier

# load word_features into memory.
word_features = Pickle_Utils.unpickle_this(r"C:\Users\parke\Documents\GitHub\NaturalLanguage\NaturalLanguage\OwnPrograms\Kaggle_Program\word_features_whenN5000_and_Tweets_all.pickle")

# load classifier_list into memory.
classifier_list =Pickle_Utils.unpickle_this(r"C:\Users\parke\Documents\GitHub\NaturalLanguage\NaturalLanguage\OwnPrograms\Kaggle_Program\list_of_fully_trained_classifiers_n_5000.pickle")
# create the Vote_Classifier based on the 9 classifers pre trained.


# submethod.
# def convert_to_vector(Tweet (string), Word_features(List) )


# what does decision_function(samples) do?


# you would also want to see the false positve and false negative rate.



