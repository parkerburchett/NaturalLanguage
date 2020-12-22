from sklearn.svm import LinearSVR
from sklearn.metrics import accuracy_score
from NaturalLanguage.custom_NLTK_Utils import dataLabeling as dl
from NaturalLanguage.custom_NLTK_Utils import AlgoEvalutationUtils as AE
from NaturalLanguage.custom_NLTK_Utils import Inital_Pickle as ip
import nltk
import datetime
import random
import numpy as np

def convert_docs_to_vectors(docs, word_features):
    """
    Parameters: docs a list of (dict, category) tuples that represent a point, output pair.
                word_features: the list of word_features to be treated as dems to classify the things

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
        True is Postive and False is Negative sentiment.
    """
    num_features = len(word_features) # constant
    vectors = np.zeros((len(docs), num_features), dtype=bool)  # this is where you store the results. Default is false
    targets = [] # a list should work here it might be slower
    for doc in range(len(docs)):
        words = nltk.word_tokenize(docs[doc][0])
        # at this point you would want to limit words by things like stop words, punctuation, or Parts of Speech
        # make sure that those words are also excluded from word_features.
        # you can also add in (if not already existing) some words you think might have a relationship with sentiment
        sentiment = str(docs[doc][1])
        vector = np.zeros(num_features, dtype=bool)

        for i in range(num_features):
            if word_features[i] in words:
                vector[i] = True

        vectors[doc] = vector
        if(sentiment =='Negative'):
            targets.append(0) #negative sentiment is a zero
        else:
            targets.append(1) # positive sentiment is a one

    #need to make targets a boolean vector
    # perhaps you need to scale(vectors) and scale(targets_as_numpy_matrix)
    targets_as_numpy_matrix = np.array(targets,dtype=bool)
    return vectors, targets_as_numpy_matrix

def convert_float_array_to_boolean(predictions):
    """
    Parameters:
        predictions is a np.array of floats all near 1 or 0. This is the output of  linearSVR().predict() function

    Returns:
        predictions_as_booleans: a np.array of booleans

      Converts based on if it is closer to 0 or 1
    """
    predictions_as_booleans = np.zeros(shape=predictions,dtype=bool)
    for pred in range(predictions):
        if predictions(pred) > .5:
            predictions_as_booleans[pred] =True

    return predictions_as_booleans



# youtube lecture
# Source: https://www.youtube.com/watch?v=KTeVOb8gaD4
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html

def train_on_kaggle_data():
    print('started')
    out = open('log_kaggle.txt','a')
    start = datetime.datetime.now()
    num_tweets = 100000
    num_features = 1000
    iters = 1000
    out.write('--------------------\nNew Model')
    out.write('When num_tweets is {}\n'.format(num_tweets))
    out.write('When num_features is {}\n'.format(num_features))
    out.write('When max_iters is {}\n'.format(iters))
    inputFile = open(r"C:\Users\parke\Documents\GitHub\NaturalLanguage\NaturalLanguage\Datasets\LabeledTweets.csv", "r")
    documents = dl.assemble_kaggle_documents(inputFile) # this shuffles it

    docs = documents[:num_tweets] # for debugging only treat the first N as sample
    all_words = []
    for d in docs:
        words = nltk.word_tokenize(d[0])
        for w in words:
            all_words.append(w.lower())

    all_wordsFRQ = nltk.FreqDist(all_words)
    word_features_as_tuples = all_wordsFRQ.most_common(num_features) # A much better version. Make sure you replace this in main. You don't need to do any sorti
    word_features=[]
    for w in word_features_as_tuples:
        word_features.append(w[0])
    out.write('Time to create word_features :{}\n'.format(str(datetime.datetime.now() - start)))
    start = datetime.datetime.now()

    print('created docs and word_features')
    ninety_percent = int(.9*num_tweets)
    training_docs = docs[:ninety_percent]
    testing_docs = docs[ninety_percent:]

    train_vectors, train_targets = convert_docs_to_vectors(training_docs,word_features)
    test_vectors, correct_test_targets = convert_docs_to_vectors(testing_docs,word_features)

    out.write('Time to created Training and Testing.  vectors and targets :{}\n'.format(str(datetime.datetime.now() - start)))
    start = datetime.datetime.now()

    print('converted to vectors')
    my_classifier = LinearSVR()
    my_classifier.fit(train_vectors,train_targets)
    out.write('Time to Train the classifier LinearSVC:{}\n'.format(str(datetime.datetime.now() - start)))

    # now you need to look at the accuracy
    print('trained classifier')
    predictions = my_classifier.predict(test_vectors)
    # I don't know how to snap these values in on direction or anthter maybe if they are closer to one or closer to zero you can phrase that like minus

    predictions_as_booleans = convert_float_array_to_boolean(predictions) # untested

    print('stop here')
    # prob correct_test_targets is a different type than predictison
    accuracy = accuracy_score(correct_test_targets, predictions_as_booleans)
    # right now the predictionars are floats betwee
    accuracy_print_out = 'TraingingSize: {} Testing size {} this model accuracy {}\n'.format(
                        len(testing_docs),len(training_docs),accuracy)
    out.write(accuracy_print_out)
    out.write("Finished Model \n--------------------\n")
    print(accuracy_print_out)
    print('got accuracy score')
    out.close()
    print('fin')
    ip.customPickle(word_features, "small_N3000_Word_features")
    ip.customPickle(my_classifier,"small_N3000_trained NF_100000")

    #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    #Use this to write an accuracy method.

train_on_kaggle_data()


