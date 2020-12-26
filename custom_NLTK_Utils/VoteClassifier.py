from nltk.classify import ClassifierI
from scipy import stats
from nltk import word_tokenize
import numpy as np


class VoteClassifier(ClassifierI):
    """
        Attributes:
            _classifiers_list: a list of trained boolean classification models.
                         They are trained on the ordering in _word_features
            _word_features: a list of word_tokenized words. They are the N most frequent words in the entire dataset.
                            this is used in _vectorize(self, raw_tweet)
            _num_features: how many unique words are treated as features

        This is a sentiment classification algorithm that classifies based on the consensus of 9 SCGClassifiers.
    """
    def __init__(self, classifier_list, word_features):
        self._classifiers_list = classifier_list
        self._word_features = word_features
        self._num_features = len(word_features)


    def vectorize(self, raw_tweet):
        """
        Description: Convert the contents of a tweet into a boolean vector

        Parameters:
            raw_tweet: A uncleaned string of the body of a tweet
        Returns:
            vector: a np boolean vector based on the contents and orientation of _word_features
            I duplicated the method from OwnPrograms.Kaggle_Program.Vectorize_Kaggle_Data.py
            I removed the
        """
        words = word_tokenize(raw_tweet)
        vector = np.zeros(self._num_features, dtype=bool) # default values of a vector are False
        for i in range(self._num_features):
            if self._word_features[i] in words:
                vector[i] = True
        # stop this and just fix in in the predict() method
        return vector


    def classify(self, raw_tweet, consensus=5): # you might want to add a show_votes method
        """
        Parameters:
            raw_tweet: a string representation of the contents of a tweet (might need to reword
            consensus: an int for the minimum number of votes need to agree classify a tweet
                Default is 5 and since there are 9 classifiers, this will classify everything.
                Increase consensus by up to 9 to only get the most clearly positive or negative sentiment.
        Returns:
            A 'Unsure', 'Negative' or 'Positive' based on the votes of the classifiers and the consensus
        """
        # convert to vector

        vector_of_tweet = self.vectorize(raw_tweet)
        vectors = np.zeros((1, self._num_features), dtype=bool)  # this is where you store the results. Default is false
        vectors.append(vector_of_tweet)
        # WORKS UP TO HERE as expected

        # get classification and num_votes.
        classification, num_votes = self.voting(vectors)

        # return a classification
        if num_votes < consensus:
            return 'Unsure' # you might want to relabel this
        else:
            if classification: # classification is a boolean
                return 'Positive'
            else:
                return 'Negative'


    def voting(self, vector):
        """
        Calcuates category and the number of classifiers that voted for that category

        Parameters:
            vector: a boolean vector representation of a tweet based on _word_features

        Returns:
            classification: A boolean representing True: Positive Sentiment and False : Negative Sentiment.
            num_votes: The number of classifiers that voted for this the most common classification
                        limited to [5,6,7,8,9] with only 9 classsifiers.
        """
        votes = []
        for c in self._classifiers_list:
            v = c.predict(vector) # this is the thing that breaks it.
            votes.append(v)
        classification = stats.mode(votes)

        return str(classification.mode[0]), classification.count


    def un_vectorize(self, vector):
        """
        Parameters:
            vector: boolean vector of a tweet.
        This takes a vector and returns a list of words that it represents.
        Because of the bag of words approach, the order and frequency of words is ignored.
        This is for debugging.
        """
        # you need to do vector[0] since vector is a 2d array of booelan vectors but only one row
        if len(vector[0]) != len(self._word_features):
            raise ValueError('vector len {}, word_features len {}'.format((len(vector[0])),len(self._word_features)))

        words =[]
        for i in range(len(vector[0])): # untested
            if vector[0][i]: # vector is an np.array of booleans
                words.append(self._word_features[i])
        return words

