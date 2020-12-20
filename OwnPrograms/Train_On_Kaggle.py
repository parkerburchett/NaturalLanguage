from sklearn.svm import LinearSVR
from NaturalLanguage.custom_NLTK_Utils import dataLabeling as dl
from NaturalLanguage.custom_NLTK_Utils import AlgoEvalutationUtils as AE
from NaturalLanguage.custom_NLTK_Utils import Inital_Pickle as ip
import nltk
import datetime
import random


# source: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html
def train_on_kaggle_data():
    start = datetime.datetime.now()
    print('started')
    inputFile = open(r"C:\Users\parke\Documents\GitHub\NaturalLanguage\NaturalLanguage\Datasets\LabeledTweets.csv", "r")
    documents = dl.assemble_kaggle_documents(inputFile)
    print('Created documents :{}'.format(str(datetime.datetime.now() - start)))
    all_words = []
    for d in documents:
        all_words.append(d[0].lower())
    all_wordsFRQ = nltk.FreqDist(all_words)
    print('Created all_words:{}'.format(str(datetime.datetime.now() - start)))


    dict(sorted(all_wordsFRQ.items(), key=lambda item: item[1]))
    word_features = list(all_words)[:5000]
    print('Created word_features :{}'.format(str(datetime.datetime.now() - start)))

    # very Hight Timecost. You might just want to pickle this after it runs.
    feature_sets = [(dl.find_Features(text, word_features), category)
                    for (text, category) in documents]

    ip.customPickle(feature_sets, "KaggleN5000_feature_sets_preShuffle")
    random.shuffle(feature_sets)

    print('Created feature_sets :{}'.format(str(datetime.datetime.now() - start)))

    train_set = AE.getTrainData(feature_sets)
    LinearSupportVectorRegression = nltk.SklearnClassifier(LinearSVR())
    LinearSupportVectorRegression.train(train_set)

    print('Trained Linear SVR :{}'.format(str(datetime.datetime.now() - start)))
    test_data =AE.getTestData(feature_sets)

    print('Accuracy: {}'.format(str(nltk.classify.accuracy(LinearSupportVectorRegression, test_data) * 100)))

    ip.customPickle(LinearSupportVectorRegression, "Kaggle_LinearSupportVectorRegression")
    ip.customPickle(word_features, "KaggleN5000_word_features")

    print('Finished :{}'.format(str(datetime.datetime.now()-start)))



train_on_kaggle_data()

