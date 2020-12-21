from sklearn.svm import LinearSVR
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
        Vectors: An np.Array of Boolean Vectors of Llngth =len(word_features).Represents each of the points in the dataset
            A tweet is mapped on to a boolean vector by looking at
        Targets: The associated Positive or Negative sentiment to with that vector

    """
    num_features = len(word_features) # constant
    vectors = np.zeros((len(docs), num_features), dtype=bool)  # this is where you store the results. Default is false
    targets = [] # a list should work here it might be slower
    for doc in range(len(docs)):
        words = nltk.word_tokenize(docs[doc][0])
        sentiment = str(docs[doc][1])
        vector = np.zeros(num_features, dtype=bool)

        for i in range(num_features):
            if word_features[i] in words:
                vector[i] = True

        vectors[doc] = vector
        targets.append(sentiment)

    return vectors, targets








# source: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html
def train_on_kaggle_data():
    start = datetime.datetime.now()
    print('started')
    inputFile = open(r"C:\Users\parke\Documents\GitHub\NaturalLanguage\NaturalLanguage\Datasets\LabeledTweets.csv", "r")
    documents = dl.assemble_kaggle_documents(inputFile) # this shuffles it
    docs = documents[:10000] # for debugging only treat the first 10k as sample
    all_words = []
    for d in docs:
        words = nltk.word_tokenize(d[0])
        for w in words:
            all_words.append(w.lower())

    all_wordsFRQ = nltk.FreqDist(all_words)

    word_features_as_tuples = all_wordsFRQ.most_common(50) # A much better version. Make sure you replace this in main. You don't need to do any sorti
    word_features=[]
    for w in word_features_as_tuples:
        word_features.append(w[0])

    print('Created word_features :{}'.format(str(datetime.datetime.now() - start)))
    vectors, targets = convert_docs_to_vectors(docs,word_features)
    # while here you need to also create the targets


    print('Created vectors and targets :{}'.format(str(datetime.datetime.now() - start)))


    for i in range(10):
        print(vectors[i])
        print(targets[i])








    #
    # # very High Time cost. You might just want to pickle this after it runs.
    # feature_sets = [(dl.find_Features(text, word_features), category)
    #                 for (text, category) in docs]
    # random.shuffle(feature_sets)
    # print('Created feature_sets :{}'.format(str(datetime.datetime.now() - start)))
    #
    # train_set = feature_sets[:9000]
    # test_set = feature_sets[9000:]
    #
    # res = convert_data_to_array(feature_sets)
    #
    #
    # my_classifier = LinearSVR()
    #
    #
    #
    #
    # data =[]
    # targets =[]



    # my_classifier.fit(vectors,cats)
    # you need to convert vectors to a 2d boolean array
    # you need to convert cats to a 1d string array


    print('Finished :{}'.format(str(datetime.datetime.now()-start)))

train_on_kaggle_data()

# when