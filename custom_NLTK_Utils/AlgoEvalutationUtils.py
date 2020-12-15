import nltk
from NaturalLanguage.custom_NLTK_Utils import AlgoParams
from NaturalLanguage.custom_NLTK_Utils import VoteClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC



def createParamList():
    shortPos = open(
        "C:/Users/parke/Documents/GitHub/NaturalLanguage/NaturalLanguage/OwnPrograms/short_reviews/shortPositive.txt",
        "r").read()
    shortNeg = open(
        "C:/Users/parke/Documents/GitHub/NaturalLanguage/NaturalLanguage/OwnPrograms/short_reviews/shortNegative.txt",
        "r").read()
    paramList = []
    for f in ("NOUN", "ADJ", "VERB", "ADV", "*"):
        paramList.append(AlgoParams.AlgoParams(True, 1000, shortPos, shortNeg, f))
    return paramList


def getTestData(feature_sets):
    N = int(len(feature_sets) * .9)
    TestingData = feature_sets[N:]
    return TestingData


def getTrainData(feature_sets):
    N = int(len(feature_sets) * .9)
    TrainingData = feature_sets[:N]
    return TrainingData


def CreateAndTrain_Classifiers(Feature_sets):
    TrainingData = getTrainData(Feature_sets)
    TrainedClassifierList = []

    c = nltk.NaiveBayesClassifier.train(TrainingData)
    TrainedClassifierList.append(c)
    NBClassifer = nltk.NaiveBayesClassifier.train(TrainingData)
    TrainedClassifierList.append(NBClassifer)

    c = SklearnClassifier(SGDClassifier())
    c.train(TrainingData)
    TrainedClassifierList.append(c)

    c = SklearnClassifier(BernoulliNB())
    c.train(TrainingData)
    TrainedClassifierList.append(c)

    c = SklearnClassifier(LinearSVC())
    c.train(TrainingData)
    TrainedClassifierList.append(c)

    c = SklearnClassifier(LogisticRegression())
    c.train(TrainingData)
    TrainedClassifierList.append(c)

    voted_classifier = VoteClassifier.VoteClassifier(TrainedClassifierList[0],
                                                     TrainedClassifierList[1],
                                                     TrainedClassifierList[2],
                                                     TrainedClassifierList[3],
                                                     TrainedClassifierList[4])
    TrainedClassifierList.append(voted_classifier)
    return TrainedClassifierList


def writeAlgoEvaluation(param, classifiers, FeatureSets):
    TestingSet = getTestData(FeatureSets)
    # change where this writes to so it writes to the cd
    with open("CurTest.txt", "a+") as out:
        out.write("----------------------------------------------\n")
        ParamDetails = ("Remove StopWords: " + str(param.the_stop) +
                        "\nN Most Frequent : " + str(param.NmostFrequent) +
                        "\nPartsOfSpeech   : " + str(param.PartsOfSpeech)
                        )
        out.write(ParamDetails)
        out.write("\nAccuracy of Naive Bayes                   :" + str(
            nltk.classify.accuracy(classifiers[0], TestingSet) * 100))
        out.write("\nAccuracy of SGD Classifiers               :" + str(
            nltk.classify.accuracy(classifiers[1], TestingSet) * 100))
        out.write("\nAccuracy of Bernoulli Naive Bayes         :" + str(
            nltk.classify.accuracy(classifiers[2], TestingSet) * 100))
        out.write("\nAccuracy of Linear Support Vector Machine :" + str(
            nltk.classify.accuracy(classifiers[3], TestingSet) * 100))
        out.write("\nAccuracy of Logistic Regression           :" + str(
            nltk.classify.accuracy(classifiers[4], TestingSet) * 100))
        out.write("\nAccuracy of Vote Classifier               :" + str(
            nltk.classify.accuracy(classifiers[5], TestingSet) * 100))
        out.write("\n----------------------------------------------\n")
