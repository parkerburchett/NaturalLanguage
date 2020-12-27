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

        if np.count_nonzero(vector_for_prediction) ==0:
            return 'Unsure' # no words are features it ought to default to unsure since you are training it on a vector of only False
        if num_votes[0] < consensus:
            return 'Unsure'  # you might want to relabel this
        else:
            # the problem is that classifcation lookes like '[False]' not a boolean False
            if classification: # classification is a is a string because I miscoded it. Forwhatever reason this now only returns 'Positive'
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
            v = c.predict(vector) # when I pass this the single word: 'cry' every algo predicts array([False])
            votes.append(v)
        classification = stats.mode(votes)  # this is a mode object
        choice = classification.mode[0]
        return choice[0], classification.count

    def get_category_votes(self, raw_tweet):
        """
            Description:
                This gets converts a raw tweet into a category, numVotes pair.
        """
        vector_of_tweet = dl.text_to_vector(raw_tweet, self._word_features)
        # vector_for_prediction needs to be 2d to work with SGDClassifier.predict(X)
        vector_for_prediction = np.zeros((1, self._num_features), dtype=bool)
        vector_for_prediction[0] = vector_of_tweet
        if np.count_nonzero(vector_for_prediction) ==0:
            return ('Unsure no known features', 9) # never seen before
        # get classification and num_votes.
        classification, num_votes = self.voting(vector_for_prediction)


        if classification:  # classification is a is a string because I miscoded it. Forwhatever reason this now only returns 'Positive'
            category = 'Positive'
        else:
            category = 'Negative'

        return category, num_votes