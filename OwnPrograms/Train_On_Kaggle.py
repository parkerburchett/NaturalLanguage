from sklearn.svm import LinearSVR
from NaturalLanguage.custom_NLTK_Utils import dataLabeling as dl
from NaturalLanguage.custom_NLTK_Utils import AlgoEvalutationUtils as AE
from NaturalLanguage.custom_NLTK_Utils import Inital_Pickle as ip
import nltk
import datetime
import random
import numpy


# source: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html
def train_on_kaggle_data():
    start = datetime.datetime.now()
    print('started')
    inputFile = open(r"C:\Users\parke\Documents\GitHub\NaturalLanguage\NaturalLanguage\Datasets\LabeledTweets.csv", "r")
    documents = dl.assemble_kaggle_documents(inputFile)
    docs = documents[:10000] # for debugging only treat the first 10k as sample
    # 10k = 14 seconds
    # 20k = 28 seconds.
    # 30k = 41 seconds
    # when len(word_features) == 1000

    # Time cost is linear with this equation: Total time  = (
    all_words = []
    for d in docs:
        words = nltk.word_tokenize(d[0])
        for w in words:
            all_words.append(w.lower())

    all_wordsFRQ = nltk.FreqDist(all_words)

    word_features_as_tuples = all_wordsFRQ.most_common(10) # A much better version. Make sure you replace this in main. You don't need to do any sorti
    word_features=[]
    for w in word_features_as_tuples:
        word_features.append(w[0])

    print('Created word_features :{}'.format(str(datetime.datetime.now() - start)))

    # very High Time cost. You might just want to pickle this after it runs.
    feature_sets = [(dl.find_Features(text, word_features), category)
                    for (text, category) in docs]
    random.shuffle(feature_sets)
    train_set = feature_sets[:9000]
    test_set = feature_sets[9000:]

    my_classifier = LinearSVR()


    myData = numpy.array(feature_sets)
    vectors =[]
    cats =[]

    for d in myData:
        vectors.append(list((d[0].items())))
        cats.append(d[1])

    my_classifier.fit(vectors,cats)


    print('Finished :{}'.format(str(datetime.datetime.now()-start)))
    # this is the syntax for fiting the algo: fit(Array maybe of vectors, Array maybe of proper outcomes.

    # you could convert the list of (dict, String) tuples into a dataset and

    # print('Accuracy: {}'.format(str(accuracy)))
    # ip.customPickle(LinearSupportVectorRegression, "Kaggle_LinearSupportVectorRegression")
    # ip.customPickle(word_features, "KaggleN5000_word_features")




train_on_kaggle_data()

# when