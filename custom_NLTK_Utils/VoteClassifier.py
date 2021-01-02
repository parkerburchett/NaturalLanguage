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
            _word_weight_dict: dynamic programming, stores the weights assigned to each word

        This is a sentiment classification algorithm that classifies based on the consensus of 9 SCGClassifiers.
    """

    def __init__(self, classifier_list, word_features, avg_accuracy):
        self._classifiers_list = classifier_list
        self._word_features = word_features
        self._num_features = len(word_features)
        self._avg_accuracy = avg_accuracy
        # self._word_weight_dict = None

    def get_classifier_list(self):
        return self._classifiers_list

    def get_num_features(self):
        return self._num_features

    def get_word_features(self):
        return self._word_features

    def get_avg_accuracy(self):
        return self._avg_accuracy

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

        classification, num_votes = self.voting(vector_for_prediction)

        if np.count_nonzero(vector_for_prediction) == 0:
            return 'Unsure'  # no words are features it ought to default to unsure since you are training it on a vector of only False
        if num_votes[0] < consensus:
            return 'Unsure'  # you might want to relabel this
        else:
            # the problem is that classifcation lookes like '[False]' not a boolean False
            if classification:  # classification is a is a string because I miscoded it. Forwhatever reason this now only returns 'Positive'
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
            v = c.predict(vector)  # when I pass this the single word: 'cry' every algo predicts array([False])
            votes.append(v)
        classification = stats.mode(votes)  # this is a mode object
        choice = classification.mode[0]
        return choice[0], classification.count

    def get_category_votes(self, raw_tweet):
        """
            Description:
                This method takes a raw tweet and returns what the the majority choice and number of votes
            Parameters:
                raw_tweet: a String for the body of a tweet
        """
        vector_of_tweet = dl.text_to_vector(raw_tweet, self._word_features)
        # vector_for_prediction needs to be 2d to work with SGDClassifier.predict(X)
        vector_for_prediction = np.zeros((1, self._num_features), dtype=bool)
        vector_for_prediction[0] = vector_of_tweet
        if np.count_nonzero(vector_for_prediction) == 0:
            # This returns if the tweet has no words  the classifiers have been trained on
            return ('Unsure no known features', 9)

        classification, num_votes = self.voting(vector_for_prediction)

        if classification:
            category = 'Positive'
        else:
            category = 'Negative'

        return category, num_votes

    def get_relevant_words(self, raw_tweet):
        """
        Description:
            Determines what words that are treated as features
        Parameters:
            raw_tweet: a String for the body of a tweet
        Returns:
            relevant_words: A list of Strings. Every word that is treated as a feature
        """
        vector_of_tweet = dl.text_to_vector(raw_tweet, self._word_features)
        relevant_words = dl.vector_to_words(vector_of_tweet, self._word_features)
        return relevant_words

    def get_relevant_words_weights(self, raw_tweet):
        """
        Description:
            Determines the relevant words and the average weight for each word.
        Parameters:
            raw_tweet:  a String for the body of a tweet
        Returns:
            words_weights: a list of Word, average_weight tuples.
        """

        relevant_words = self.get_relevant_words(raw_tweet)
        weight_dictionary = self.get_avg_weight_dictionary()

        words_weights = []
        for w in relevant_words:
            word_weight_pair = (w, weight_dictionary[w])
            words_weights.append(word_weight_pair)

        return words_weights

    def get_avg_weight_dictionary(self):
        """
        Description:
            This method returns a dictionary object for each word: avg weight.
            if it does not exist it stores it in a self_word_weightdict
        """
        weights_list = []
        for a_classifier in self._classifiers_list:
            weights_list.append(a_classifier.coef_[0])

        avg_word_weights = {}

        for word_index in range(len(self._word_features)):
            all_weights_for_word = []
            for weight in weights_list:
                all_weights_for_word.append(weight[word_index])
            avg_word_weights[self._word_features[word_index]] = np.average(all_weights_for_word)

        return avg_word_weights

    def get_scores(self, raw_tweet):
        """
        Description:
            This method takes in a string and returns a list scores that each classifiers computes using
            SGDClassifier.decision_function(x)
            1 corresponds to Positive
            0 corresponds to Negative

        Parameter:
            raw_tweet: A string
        Returns:
            scores: a list of scores from each of the classifiers.
        """
        scores = []
        vector_of_tweet = dl.text_to_vector(raw_tweet, self._word_features)
        # vector_for_prediction needs to be 2d to work with SGDClassifier.predict(X)
        vector_for_prediction = np.zeros((1, self._num_features), dtype=bool)
        vector_for_prediction[0] = vector_of_tweet
        for c in self._classifiers_list:
            scores.append(c.decision_function(vector_for_prediction)[0])

        return scores

    def explain_choice(self, raw_tweet):
        """
        Description:
            Stitches together the reasoning and stats that made for why the raw_tweet was classified as it was
        Parameters:
            raw_tweet: a String. Designed to work with the twitter stream
        Returns
            write_up: a long String that explains why the classifier made that choice.
        """
        category_choice = self.classify(raw_tweet)
        avg_score = np.average(self.get_scores(raw_tweet))
        write_up = 'The tweet:\n{}\nIs:\n{}\nAverage score:\n{}\n'.format(raw_tweet,category_choice,round(avg_score,4))
        words_that_matter = self.get_relevant_words(raw_tweet)
        write_up = write_up + 'This decision was based on these {} words:\n{}\n'.format(len(words_that_matter),words_that_matter)
        word_weights = self.get_relevant_words_weights(raw_tweet)
        # sort word_weights by the absolute value of their weights
        word_weights = sorted(word_weights, key=lambda x: abs(x[1]),reverse=True)

        word_weights_to_write =[]
        try: # this is for when there are less than 5 words as features.
            for w in range(5):
                pair = word_weights[w][0],round(word_weights[w][1],4)
                word_weights_to_write.append(pair)
        except:
            print('you did not have 5 features')

        write_up = write_up + 'These were the {} words with the most influence on the outcome:\n{}\n'.format(len(word_weights_to_write),word_weights_to_write)

        return write_up