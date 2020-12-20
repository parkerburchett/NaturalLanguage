from nltk import SklearnClassifier
from sklearn.svm import LinearSVC
from NaturalLanguage.custom_NLTK_Utils import dataLabeling as dl
from NaturalLanguage.custom_NLTK_Utils import AlgoEvalutationUtils as AE
import nltk
import datetime

def train_on_kaggle_data():
    start = datetime.datetime.now()
    print('started')
    inputFile = open(r"C:\Users\parke\Documents\GitHub\NaturalLanguage\NaturalLanguage\Datasets\LabeledTweets.csv", "r")
    documents = dl.assemble_kaggle_documents(inputFile)

    print('Created Documents :{}'.format(str(datetime.datetime.now() - start)))
    feature_sets = dl.kaggle_create_feature_sets(documents,5000)
    
    print('Created Feature_sets :{}'.format(str(datetime.datetime.now() - start)))

    train_set = AE.getTrainData(feature_sets)
    LinearSVCClassifier = SklearnClassifier(LinearSVC())
    LinearSVCClassifier.train(train_set)

    print('Trained Linear SCV :{}'.format(str(datetime.datetime.now() - start)))
    test_data =AE.getTestData(feature_sets)
    print('Accuracy: {}'.format(str(nltk.classify.accuracy(LinearSVCClassifier, test_data) * 100)))

    print('Finished :{}'.format(str(datetime.datetime.now()-start)))

train_on_kaggle_data()

