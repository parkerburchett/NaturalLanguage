from nltk.classify import ClassifierI
from scipy import stats
from NaturalLanguage.custom_NLTK_Utils import dataLabeling as dl
from nltk import word_tokenize
import numpy as np


class VoteClassifier(ClassifierI):
    """
        Attributes:
            _classifiers_list: a list of trained boolean classification models.
                         They are trained on the ordering in _word_features
            _word_features: a list of the words treated as features in this model
            _num_features: how many unique words are treated as features
            _avg_accuracy: the average accuracy of the 9 classifiers

        This is a sentiment classification algorithm that classifies based on the consensus of 9 SCGClassifiers.
    """

    def __init__(self, classifier_list, word_features,avg_accuracy):
        self._classifiers_list = classifier_list
        self._word_features = word_features
        self._num_features = len(word_features)
        self._avg_accuracy = avg_accuracy # replace this with the list of accuracy scores increase in order

    def get_num_features(self):
        return self._num_features

    def classify(self, raw_tweet, consensus=5):  # you might want to add a show_votes method
        """
        Parameters:
            raw_tweet: a string representation of the contents of a tweet (might need to reword
            consensus: an int for the minimum number of votes need to agree classify a tweet
                Default is 5 and since there are 9 classifiers, this will classify everything.
                Increase consensus by up to 9 to only get the most clearly positive or negative sentiment.
        Returns:
            A 'Unsure', 'Negative' or 'Positive' based on the votes of the classifiers and the consensus
        """
        vector_of_tweet = dl.text_to_vector(raw_tweet, self._word_features)
        # vector_for_prediction needs to be 2d to work with SGDClassifier.predict(X)
        vector_for_prediction = np.zeros((1, self._num_features), dtype=bool)
        vector_for_prediction[0] = vector_of_tweet

        # get classification and num_votes.
        classification, num_votes = self.voting(vector_for_prediction)

        # return a classification
        if num_votes < consensus:
            return 'Unsure'  # you might want to relabel this
        else:
            if classification:  # classification is a boolean. Forwhatever reason this now only returns 'Positive'
                return 'Positive'
            else:
                return 'Negative'

    def voting(self, vector):
        """
        Calculates category and the number of classifiers that voted for that category
        Parameters:
            vector: a boolean vector representation of a tweet based on _word_features
        Returns:
            classification: A boolean representing True: Positive Sentiment and False : Negative Sentiment.
            num_votes: The number of classifiers that voted for this the most common classification
                        limited to [5,6,7,8,9] with only 9 classifiers.
        """
        votes = []
        for c in self._classifiers_list:
            v = c.predict(vector)  # this is the thing that breaks it.
            votes.append(v)
        classification = stats.mode(votes)  # this is a mode object
        return str(classification.mode[0]), classification.count
