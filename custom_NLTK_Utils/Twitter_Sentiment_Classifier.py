"""
This module will takes a tweet as a string and depending on the method.

Assigns a score. (based on the average score of the 9 classifiers)

Assigns a Classification of Positive, Negative or Unsure.

I might update it later but this method will use the SGD models trained in Kaggle_Program.Train_Kaggle.py
When Num_Features=5000.

This is slightly faster, but otherwise indistinguishable,  from the accuracy of Num_Features =10000
"""

from NaturalLanguage.custom_NLTK_Utils import Pickle_Utils
from NaturalLanguage.custom_NLTK_Utils import VoteClassifier

# load word_features into memory.
word_features = Pickle_Utils.unpickle_this(r"C:\Users\parke\Documents\GitHub\NaturalLanguage\NaturalLanguage\OwnPrograms\Kaggle_Program\word_features_whenN10000_and_Tweets_all.pickle")

# load classifier_list into memory.
classifier_list = Pickle_Utils.unpickle_this(r"C:\Users\parke\Documents\GitHub\NaturalLanguage\NaturalLanguage\OwnPrograms\Kaggle_Program\list_of_fully_trained_classifiers_n10000.pickle")
# create the Vote_Classifier based on the 9 classifoiers pre trained.

# create a VoteClassifier
the_vote_classifier= VoteClassifier.VoteClassifier(classifier_list, word_features)

# problem: Every algo Cla
def get_sentiment(raw_tweet, consensus_choice=(len(classifier_list)+1)/2):
        category = the_vote_classifier.classify(raw_tweet, consensus=consensus_choice)
        return category


# Known problem: it is classifying things that should be negative as postive.
for i in (5,6,7,8,9):
    tester = "Hollis' death scene will hurt me severely to watch on film  wry is directors cut not out now?"
    tester2 ="hate"
    ans =get_sentiment(tester2,i)
    print('With {} Votes it classifies as {}'.format(i,ans))


# you would also want to see the false positve and false negative rate.



