from sklearn.svm import LinearSVR
from sklearn.metrics import accuracy_score
from NaturalLanguage.custom_NLTK_Utils import dataLabeling as dl
from NaturalLanguage.custom_NLTK_Utils import AlgoEvalutationUtils as AE
from NaturalLanguage.custom_NLTK_Utils import Pickle_Utils as ip
import nltk
import datetime
import random
import numpy as np

def convert_docs_to_vectors(docs, word_features, num_features):
    """
    Parameters: docs a list of (dict, category) tuples that represent a point, output pair.
                word_features: the list of word_features to be treated as dems to classify the things
                num_features: how many unique words are treated as features

    Returns:
        Vectors: An np.array of Boolean Vectors of length =len(word_features).
        It represents each of the tweets in the dataset
            A tweet is mapped on to a boolean vector by looking at what words are contained in it.

            For example, if the "great" is in word_features at index 12.
            Every tweet that contains "great" will have a True at index 12 and every tweet without "great" will have a false

            The way I think about it is: each tweet is converted into a point in a num_features Dimensional space.
            Then the algos now that they have the a collection of point -> output pairs do thier best to draw a hyperplane
            A num_features -1 dimensional subspace such that the it divides the positive and negative tweets the best it can.

        Targets: The associated Positive or Negative sentiment to with that Vector of the same index.
        True is Positive and False is Negative sentiment.
    """
    if num_features != len(word_features):
        raise ValueError('num_features different than the len of word_features')


    vectors = np.zeros((len(docs), num_features), dtype=bool)  # this is where you store the results. Default is false
    targets = np.zeros(shape=len(docs),dtype=bool)
    for doc in range(len(docs)):
        words = nltk.word_tokenize(docs[doc][0])
        sentiment = str(docs[doc][1])
        vector = np.zeros(num_features, dtype=bool)
        for i in range(num_features):
            if word_features[i] in words:
                vector[i] = True

        vectors[doc] = vector
        if sentiment == 'Positive':
            targets[doc] = True

    return vectors, targets
# not used. might want to remove
def convert_float_array_to_boolean(predictions): # not used
    """
    Parameters:
        predictions is a np.array of floats all near 1 or 0. This is the output of  linearSVR().predict() function

    Returns:
        predictions_as_booleans: a np.array of booleans

      Converts based on if it is closer to 0 or 1
    """
    predictions_as_booleans = np.zeros(shape=(len(predictions)),dtype=bool)
    for pred in range(len(predictions)):
        if predictions[pred] > .5:
            predictions_as_booleans[pred] =True

    return predictions_as_booleans



# youtube lecture
# Source: https://www.youtube.com/watch?v=KTeVOb8gaD4
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html

def train_on_kaggle_data():
    print('started')
    out = open('log_kaggle.txt','a')
    start = datetime.datetime.now()
    num_tweets = 'all'
    num_features = 6000
    # you should include sections here to print out what chunks you have done
    out.write('--------------------\n\nNew Model\n')
    out.write('When num_tweets is {}\n'.format(num_tweets))
    out.write('When num_features is {}\n'.format(num_features))
    inputFile = open(r"C:\Users\parke\Documents\GitHub\NaturalLanguage\NaturalLanguage\Datasets\LabeledTweets.csv", "r")

    documents = dl.assemble_kaggle_documents(inputFile) # this shuffles and has a low time cost

    docs = documents # for debugging only treat the first N as sample
    all_words = []
    for d in docs:
        words = nltk.word_tokenize(d[0])
        for w in words:
            all_words.append(w.lower())

    all_wordsFRQ = nltk.FreqDist(all_words)
    word_features_as_tuples = all_wordsFRQ.most_common(num_features) # this is so clean

    word_features=[]
    for w in word_features_as_tuples:
        word_features.append(w[0])
    out.write('Time to create word_features :{}\n'.format(str(datetime.datetime.now() - start)))

    sorted(word_features) # this is to assure that the the classifiers are all in the right order

    start = datetime.datetime.now()

    print('created docs and word_features')
    ip.pickle_this(docs, "ordering_of_docs_when_N{}".format(num_features))
    ip.pickle_this(word_features, "word_features_when_N{}".format(num_features))

    # expensive
    vectors, targets = convert_docs_to_vectors(docs,word_features, num_features) # takes 2 hours
    #expensive

    ip.pickle_this(vectors,'vectors_when_N{}'.format(num_features))
    ip.pickle_this(targets, 'targets_N{}'.format(num_features))

    out.write('Time to created Training and Testing.  vectors and targets :{}\n'.format(str(datetime.datetime.now() - start)))
    print('Created and pickled Vectors and targets')
    # # when you try and fit the algo you don't have enough memory to do that. Either use a different classifier
    # # you need to find a way to make the linear SVR not allocate as float64



# you will need to use SGDClassifier
    # # inside of #fit() set dtype = bool?
    # my_classifier.fit(train_vectors,train_targets)
    #
    #
    # out.write('Time to Train the classifier LinearSVC:{}\n'.format(str(datetime.datetime.now() - start)))
    #
    # # now you need to look at the accuracy
    # print('trained classifier')
    # predictions = my_classifier.predict(test_vectors)
    # # Need to snap the predictions an array of floats into an array of booleans
    # # use https://scikit-learn.org/stable/auto_examples/applications/plot_out_of_core_classification.html
    #
    # # you will want ot use partial_fit()
    #
    #
    # # https://stackoverflow.com/questions/20643300/training-sgdregressor-on-a-dataset-in-chunks
    #
    # # idea break the Trainingset (Vector:Target) into 1 percent chunks. do partial_fit()
    # # MemoryError: Unable to allocate 67.1 GiB for an array with shape (900000, 10000) and data type float64
    # #
    # # predictions_as_booleans = convert_float_array_to_boolean(predictions) # untested
    # # accuracy = accuracy_score(correct_test_targets, predictions_as_booleans)
    # #
    # # accuracy_print_out = 'Training Size: {} Testing Size {} This model accuracy {}\n'.format(
    # #                     len(testing_docs),len(training_docs),accuracy)
    # # out.write(accuracy_print_out)
    # # out.write("Finished Model \n--------------------\n")
    # # print(accuracy_print_out)
    # # print('got accuracy score')
    # # out.close()
    # # print('fin')
    # # ip.customPickle(word_features, "word_features_Feats{}_Tweets{}".format(num_features,num_tweets))
    # # ip.customPickle(my_classifier,"classifier_Feats{}_Tweets{}".format(num_features,num_tweets))

    #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    #Use this to write an accuracy method.

train_on_kaggle_data()


