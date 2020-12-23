"""
This Module  will use the partial_fit() method on the SGDClassifier() to develop a model
for analyzing Twitter sentiment. It will also show the relationship between training size and accuracy.

Initially you have 5000 words as features, without any limitations. This includes punctuation, and stopwords


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
    This method takes the first N (default: N=100,000) tweet: sentiment pairs
    and uses partial fit to train a SGDClassifier

    """
    my_classifier = SGDClassifier()
    all_classes = np.unique(targets)  # [True, False]
    my_classifier.partial_fit(vectors[:N], targets[:N], all_classes)  # you only need to include targets once
    return my_classifier


def get_data_from_pickle():
    """
    Returns the entire Tweet_as Vector : targets pair
    """
    vectors = Pickle_Utils.unpickle_this(r"vectors_when_N5000.pickle")
    targets = Pickle_Utils.unpickle_this(r"targets_N5000.pickle")
    return vectors, targets


def train_more(my_classifier, vectors, targets, numTrained, new_tweets=10000):
    """
    Parameters:
        my_classifier: A  trained SGDClassifier()
        vectors: the vectors, a NP array of Booleans
        targets: the target boolean. False = Negative | True = Positive
        numTrained: the index of the number of tweets you have already trained on.
        new_tweets: the number of new tweets to train on.
    Returns:
        Nothing, it just trains my_classifier
    """
    end_trained = numTrained + new_tweets
    my_classifier.partial_fit(vectors[numTrained:end_trained], targets[numTrained:end_trained])


def log_accuracy(my_classifier, vectors, targets, num_trained, log_file,
                 sample_size=10000, sections=10 ):
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
        cur_index =int(cur_index+section_size)

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
        print(to_write) # debugging, remove this later
        log_file.write(to_write)


def train_and_log(my_classifier, vectors, targets,global_start):
    """
        Parameters:
        my_classifier: A trained SGDClassifier()
        vectors: the vectors, a NP array of np boolean arrays
        targets: np.array of target booleans.
        global_start: the time at the start of this array. This is to have visual progress of movement


    This stitches the train and log methods together around a for loop
    """
    # I don't know the upperbound for how much I can run this I am running it 10 times to check
    log_file = open('./SGD_scores.csv', "w")
    log_file.write('training_size, sample_size, accuracy\n')
    num_trained = 10000
    try:
        for i in range(1000):
            new_tweets = 10000
            train_more(my_classifier, vectors, targets, num_trained, new_tweets)
            log_accuracy(my_classifier, vectors, targets, num_trained, log_file)
            num_trained = num_trained + new_tweets
            print('finTrainTestSplit:{}'.format(str(datetime.datetime.now() - global_start)))
    except:
        print('you go an error on this run {}'.format(str(i)))
def main():
    global_start = datetime.datetime.now()
    print('start')
    vectors, targets = get_data_from_pickle()
    my_classifier = train_intial_classifier(vectors, targets)
    train_and_log(my_classifier,vectors,targets,global_start)

    print('fin:{}'.format(str(datetime.datetime.now() - global_start)))

main()
