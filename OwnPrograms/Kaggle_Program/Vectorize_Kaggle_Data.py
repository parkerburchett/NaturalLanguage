"""
This is the model I am using to the tweet:sentiment pairs into boolean vector representation


"""



from NaturalLanguage.custom_NLTK_Utils import dataLabeling as dl
from NaturalLanguage.custom_NLTK_Utils import Pickle_Utils as ip
import nltk
import datetime
import numpy as np

def convert_docs_to_vectors(docs, word_features, num_features):
    """
    Parameters: docs a list of (dict, category) tuples that represent a point, output pair.
                word_features: the list of word_features to be treated as dems to classify the things
                num_features: how many unique words are treated as features

    Returns:
        Vectors: An np.array of Boolean vectors of size num_features
        It represents each of the tweets in the dataset
            A tweet is mapped on to a boolean vector by looking at what words are contained in it.

            For example, if the "great" is in word_features at index 12.
            Every tweet that contains "great" will have a True at index 12
            and every tweet without "great" will have a False

            The way I think about it is: each tweet is converted into a point in a num_features Dimensional space.
            Then the algos now that they have the a collection of point -> output pairs do their best
            to draw a hyperplane (a size num_features-1 dimensional subspace)
            such that the it distance of each of each point to .

        Targets: The associated Positive or Negative sentiment to with that Vector of the same index.
        True is Positive and False is Negative sentiment.
    """
    if num_features != len(word_features):
        raise ValueError('num_features different than the len of word_features')

    vectors = np.zeros((len(docs), num_features), dtype=bool)  # this is where you store the results. Default is false
    targets = np.zeros(shape=len(docs), dtype=bool)
    for doc in range(len(docs)):
        if doc % 10000 ==0:
            print('Converted this many docs {}'.format((doc)))# this is just to get a output that progress is begin made
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
def convert_float_array_to_boolean(predictions):  # not used
    """
    Parameters:
        predictions is a np.array of floats all near 1 or 0. This is the output of  linearSVR().predict() function

    Returns:
        predictions_as_booleans: a np.array of booleans

      Converts based on if it is closer to 0 or 1
    """
    predictions_as_booleans = np.zeros(shape=(len(predictions)), dtype=bool)
    for pred in range(len(predictions)):
        if predictions[pred] > .5:
            predictions_as_booleans[pred] = True

    return predictions_as_booleans


def get_word_features(num_features, default=True, remove_stopwords=False):
    if default:
        input_file = open(r"C:\Users\parke\Documents\GitHub\NaturalLanguage\NaturalLanguage\Datasets\LabeledTweets.csv", "r")
        documents = dl.assemble_kaggle_documents(input_file)  # this shuffles and has a low time cost
        input_file.close()
        return dl.kaggle_assemble_word_features(documents, num_features,remove_stopwords)
    else:
        input_file = open(r"C:\Users\parke\Documents\GitHub\NaturalLanguage\NaturalLanguage\Datasets\tinydocs.txt",
                          "r")
        documents = dl.assemble_kaggle_documents(input_file)  # this shuffles and has a low time cost
        input_file.close()
        return dl.kaggle_assemble_word_features(documents, num_features,remove_stopwords)

def create_vectors_targets(num_features, default =True, remove_stopwords1=False):
    print('started create_vectors_targets')
    out = open('log_kaggle.txt', 'a')
    start = datetime.datetime.now()
    out.write('--------------------\nNew Model\n')
    out.write('When num_features is {}\n'.format(num_features))
    if default:
        input_file = open(r"C:\Users\parke\Documents\GitHub\NaturalLanguage\NaturalLanguage\Datasets\LabeledTweets.csv", "r")
    else:
        input_file = open(r"C:\Users\parke\Documents\GitHub\NaturalLanguage\NaturalLanguage\Datasets\tinydocs.txt",
                            "r")

    documents = dl.assemble_kaggle_documents(input_file)  # this shuffles and has a low time cost

    word_features = dl.kaggle_assemble_word_features(documents, num_features, remove_stopwords=remove_stopwords1)

    out.write('Create word_features :{}\n'.format(str(datetime.datetime.now() - start)))

    start = datetime.datetime.now()

    print('created docs and word_features')
    #ip.pickle_this(word_features, "word_features_when_N{}".format(num_features))

    # expensive
    vectors, targets = convert_docs_to_vectors(documents, word_features, num_features)  # takes 2 hours

   # ip.pickle_this(vectors, 'vectors_when_N{}'.format(num_features))
   # ip.pickle_this(targets, 'targets_N{}'.format(num_features))
    out.write(
        'Time to create vectors and target:{}\n'.format(str(datetime.datetime.now() - start)))
    print('Created and pickled vectors and targets when N = {}'.format(num_features))

    return vectors, targets

def create_lemma_vectors_targets(num_features=2000):
    input_file = open(r"C:\Users\parke\Documents\GitHub\NaturalLanguage\NaturalLanguage\Datasets\LabeledTweets.csv",
                      "r")
    documents = dl.assemble_lemma_documents(input_file)
    word_features = dl.assemble_lemma_word_features(input_file)
    vectors, targets = convert_docs_to_vectors(documents,word_features,num_features)
    return vectors, targets # might be redundent

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

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
# Use this to write an accuracy method.


