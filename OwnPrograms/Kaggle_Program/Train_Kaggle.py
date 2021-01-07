"""
This Module  will use the partial_fit() method on the SGDClassifier() to develop a model
for analyzing Twitter sentiment. It will also show the relationship between training size and accuracy.

Initially you have N words as features, without any limitations. This includes punctuation, and stopwords

When running this I already have vectorized representations of the text data and sentiment. 

I would like to see how the accuracy of the classifier is impacted by adding more data. 
I want this data in do be as a spread of a large number of tests. 

EG get Accuracy scores for the classifier on 100 Random samples of N correct vs incorrect. 
I need to get a picture of the degree of variance in Accuracy as test size changes.

I also need to know the time cost of testing different sample size.

First step I need to train the classifier on 100,000 Examples as a base line. This will be used to determine the 
time cost of a single classification. 

To this end after I will break up the X,Y pairs into N% chunks to train the classifier on. 
Then after each round I will check the accuracy on a random sample of F tweets.




"""
from NaturalLanguage.custom_NLTK_Utils import Pickle_Utils
from sklearn.linear_model import SGDClassifier
from NaturalLanguage.custom_NLTK_Utils import VoteClassifier
from NaturalLanguage.OwnPrograms.Kaggle_Program import Vectorize_Kaggle_Data
from NaturalLanguage.custom_NLTK_Utils import dataLabeling as dl
from nltk.classify import NaiveBayesClassifier
import nltk

import numpy as np
import datetime


def train_intial_classifier(vectors, targets, N=10000):
    """
    Parameters:
        vectors: the vectors, a NP array of Booleans
        targets: the target boolean. False = Negative | True = Positive
        N: the initial number of tweets to use on the model
    Returns:
        A trained SGDClassifier()
    This method takes the first N (default: N=10,000) tweet: sentiment pairs
    and uses partial fit to train a SGDClassifier

    """
    my_classifier = SGDClassifier()
    all_classes = np.unique(targets)  # [True, False]
    my_classifier.partial_fit(vectors[:N], targets[:N], all_classes)  # you only need to include targets once
    return my_classifier


def get_data_from_pickle(num_features=10000):
    """
        Description:
            Loads vectors and targets into memory
        Parameter:
            num_features: what vector, target pair to use.
        Returns:
            vectors: an array of boolean vectors
            targets: the corresponding category the vector belongs to.
    """
    if num_features == 10000:  # untested
        vectors = Pickle_Utils.unpickle_this(r"vectors_when_N10000.pickle")
        targets = Pickle_Utils.unpickle_this(r"targets_N10000.pickle")
    else:
        vectors = Pickle_Utils.unpickle_this(
            r"vectors_when_N6000.pickle")  # these names are bad, you need to verify them
        targets = Pickle_Utils.unpickle_this(r"targets_N6000.pickle")
    return vectors, targets


def train_more(my_classifier, vectors, targets, numTrained, new_tweets=10000):
    """
    Parameters:
        my_classifier: A  trained SGDClassifier()
        vectors: the vectors, a NP array of Booleans
        targets: the target boolean. False = Negative | True = Positive
        numTrained: the index of the number of tweets you have already trained on.
        new_tweets: the number of new tweets to train on.
    """
    end_trained = numTrained + new_tweets
    my_classifier.partial_fit(vectors[numTrained:end_trained], targets[numTrained:end_trained])


def log_accuracy(my_classifier, vectors, targets, num_trained, log_file,
                 sample_size=10000, sections=10):
    """
    Parameters:
        my_classifier: A trained SGDClassifier()
        vectors: the vectors, a NP array of np boolean arrays
        targets: np.array of target booleans.
        num_trained: the index of the number of tweets you have already trained on.
        log_file: A open file object you are appending to
        sample_size: the number of new tweets to predict and record the accuracy of the classifier
        sections: the size of each sample to get an accuracy score on.
    """
    sample_indexes = [num_trained]
    section_size = sample_size / sections
    cur_index = num_trained
    for s in range(sections):
        sample_indexes.append(int(cur_index + section_size))
        cur_index = int(cur_index + section_size)

    # at the end of this look sample_indexes should look like:
    # [10000, 11000, 12000 , .... , 18000, 19000, 20000]
    # you will use N and N+1 for the indexes of the sample

    for index in range(len(sample_indexes) - 1):
        start_sample = sample_indexes[index]
        end_sample = sample_indexes[index + 1]
        accuracy = my_classifier.score(vectors[start_sample:end_sample], targets[start_sample:end_sample])
        # format to write in log file
        # ("numTweets trained on": int, "sample_section_size": int, Accuracy Score: float)
        to_write = "{},{},{}\n".format(str(num_trained), str(section_size), str(accuracy))
        print(to_write)  # debugging, remove this later
        log_file.write(to_write)


def train_and_log(my_classifier, vectors, targets, global_start):
    """
        Parameters:
        my_classifier: A trained SGDClassifier()
        vectors: the vectors, a NP array of np boolean arrays
        targets: np.array of target booleans.
        global_start: the time at the start of this array. This is to have visual progress of movement


    This stitches the train and log methods together around a for loop
    """
    # I don't know the upperbound for how much I can run this I am running it 10 times to check
    log_file = open('./SGD_scores_N10000_max.csv', "w")
    log_file.write('training_size, sample_size, accuracy\n')
    num_trained = 10000
    try:
        for i in range(1000):  # this trains until it runs out of data
            new_tweets = 10000
            train_more(my_classifier, vectors, targets, num_trained, new_tweets)
            log_accuracy(my_classifier, vectors, targets, num_trained, log_file)
            num_trained = num_trained + new_tweets
            print('finTrainTestSplit:{}'.format(str(datetime.datetime.now() - global_start)))
    except:
        print('you go an error on this run {}'.format(str(i)))


def create_classifier_list(vectors, targets, starting_size=10000, block_size=10000):
    """
    Parameters:
        vectors: 2d boolean np.array
        targets: 1d boolean np.array
        starting_size: the size of the data to call the first intial training on
        block_size: the size of the data to do each partial_fit call on.

    Returns 9 instances of a fulling trained SGDClassifier each with slightly different parameters.
    """
    global_start = datetime.datetime.now()

    # create 9 classifiers, each with different params options.
    # I am choosing the vary the loss function, and the penalty.
    # depending on time cost and accuracy I might use a different model

    classifier_list = [SGDClassifier(loss='hinge', penalty='l1'),
                       SGDClassifier(loss='log', penalty='l1'),
                       SGDClassifier(loss='modified_huber', penalty='l1'),
                       SGDClassifier(loss='hinge', penalty='l2'),
                       SGDClassifier(loss='log', penalty='l2'),
                       SGDClassifier(loss='modified_huber', penalty='l2'),
                       SGDClassifier(loss='hinge', penalty='elasticnet'),
                       SGDClassifier(loss='log', penalty='elasticnet'),
                       SGDClassifier(loss='modified_huber', penalty='elasticnet')]

    # Initial Training:
    for a_classifier in classifier_list:
        a_classifier.partial_fit(vectors[:starting_size], targets[:starting_size], classes=np.unique(targets))

    # Subsequent training
    try:
        num_trained = starting_size
        # each of these takes ~3 seconds.
        for i in range(119):  # this trains until it runs out of data
            local_start = datetime.datetime.now()
            for a_classifier in classifier_list:
                train_more(a_classifier, vectors, targets, num_trained, block_size)
            # only increment this after you train every algo
            num_trained = num_trained + block_size
            print('Call {} took :{}'.format(i, str(datetime.datetime.now() - local_start)))
    except:
        print('broke on this call: {}. The total time was {}'.format(i, str(datetime.datetime.now() - global_start)))

    return classifier_list

# this works to train SCG classifiers.
def train_create_VoteClassifier(the_num_features=5000, remove_stopwords=False):
    print('started train_create_VoteClassifier')
    start = datetime.datetime.now()
    outer_start = start
    vectors, targets = Vectorize_Kaggle_Data.create_vectors_targets(
        num_features=the_num_features,
        remove_stopwords1=remove_stopwords)

    word_features_local = Vectorize_Kaggle_Data.get_word_features(num_features=the_num_features,
                                                                  remove_stopwords=remove_stopwords)

    print('Time to get vectors, targets: {}'.format(str(datetime.datetime.now() - start)))
    start = datetime.datetime.now()

    classifier_list = create_classifier_list(vectors, targets)

    print('Time to train_classifiers: {}'.format(str(datetime.datetime.now() - start)))
    start = datetime.datetime.now()

    accuracy_scores = []
    for a_classifier in classifier_list:
        # this is spot check for accuracy
        accuracy = a_classifier.score(vectors[(120 * 10000):], targets[(120 * 10000):])
        accuracy_scores.append(accuracy)
        print('a classifiers accuracy : {}'.format(accuracy))



    finished_vote_classifier = VoteClassifier.VoteClassifier(classifier_list,
                                                             word_features=word_features_local,
                                                             avg_accuracy=accuracy_scores)

    print('Time to create VoteClassifier: {}'.format(str(datetime.datetime.now() - start)))
    try:
        Pickle_Utils.pickle_this(finished_vote_classifier, 'remove_stopwords_TrainedVoteClassifier_N{}'.format(the_num_features))
        print('TotalTime when N={} : {}'.format(the_num_features), str(datetime.datetime.now() - outer_start))
    except:
        print('broke on pickling the VoteClassifiers')

    print(str(datetime.datetime.now() - outer_start))

def train_lemma_naive_bayes():
    """
    This is training a naive bayes on the lemma version of the tweets.
    """
    print('start')
    start = datetime.datetime.now()
    feature_sets = Vectorize_Kaggle_Data.create_lemma_feature_sets(num_features=2000, smaller_data_size=False)


    print('created feature sets {}'.format(str(datetime.datetime.now() -start)))
    N =int(len(feature_sets)*.9)

    my_naive_bayes = NaiveBayesClassifier.train(feature_sets[:N])
    print( "Classifier accuracy percent:" ,(nltk.classify.accuracy(my_naive_bayes, feature_sets[N:])) *100 )
    print(my_naive_bayes.most_informative_features(10))
    print('Trained NB Classifier {}'.format(str(datetime.datetime.now() - start)))
    print('fin')

    return my_naive_bayes



def main():
    #train_create_VoteClassifier(the_num_features=1000, remove_stopwords=True)

    my_naive_bayes = train_lemma_naive_bayes()
    Pickle_Utils.pickle_this(my_naive_bayes,'lemma_naive_bayes_N_2000')
    print('you have picked the lemmatized version of the NB')

main()