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
    for f in ("NOUN","VERB","ADV","ADJ", "DET","AUX" "PUNCT", "*"):
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


def createAndTrain_Classifiers(Feature_sets):
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


def writeAlgoEvaluation(param, classifiers, FeatureSets, fileName = "./test_output", humanReadable=True):
    """
    Paramters:
    param: AlgoParam
    classifiers: A list of trained classifiers
    FeatureSets : a list of FeatureSets see custom_NLTK_utils.dataLabeling.py for details
    fileName : OPTIONAL, specifies where you to print the output
    humanReadable : If true: Write as a human readable ELSE: write as .csv.
    use humanReadable=False to export ot graph results
    """
    if humanReadable:
        testing_set = getTestData(FeatureSets)
        with open(fileName + ".txt", "a+") as out:
            out.write("----------------------------------------------\n")
            ParamDetails = ("Remove StopWords: " + str(param.the_stop) +
                            "\nN Most Frequent : " + str(param.NmostFrequent) +
                            "\nPartsOfSpeech   : " + str(param.PartsOfSpeech)
                            )
            out.write(ParamDetails)
            out.write("\nAccuracy of Naive Bayes                   :" + str(
                nltk.classify.accuracy(classifiers[0], testing_set) * 100))
            out.write("\nAccuracy of SGD Classifiers               :" + str(
                nltk.classify.accuracy(classifiers[1], testing_set) * 100))
            out.write("\nAccuracy of Bernoulli Naive Bayes         :" + str(
                nltk.classify.accuracy(classifiers[2], testing_set) * 100))
            out.write("\nAccuracy of Linear Support Vector Machine :" + str(
                nltk.classify.accuracy(classifiers[3], testing_set) * 100))
            out.write("\nAccuracy of Logistic Regression           :" + str(
                nltk.classify.accuracy(classifiers[4], testing_set) * 100))
            out.write("\nAccuracy of Vote Classifier               :" + str(
                nltk.classify.accuracy(classifiers[5], testing_set) * 100))
            out.write("\n----------------------------------------------\n")
    else:
        testing_set = getTestData(FeatureSets)
        with open(fileName +".csv", "a+") as out:
            N =","+str(param.NmostFrequent)
            out.write("\nAccuracy of Naive Bayes,"
                        +str(nltk.classify.accuracy(classifiers[0], testing_set) * 100) + N)
            out.write("\nAccuracy of SGD Classifier,"
                        +str(nltk.classify.accuracy(classifiers[1], testing_set) * 100) + N)
            out.write("\nBernoulli Naive Bayes,"
                        +str(nltk.classify.accuracy(classifiers[2], testing_set) * 100) + N)
            out.write("\nLinear Support Vector Machine,"
                        +str(nltk.classify.accuracy(classifiers[3], testing_set) * 100) + N)
            out.write("\nLogistic Regression,"
                        +str(nltk.classify.accuracy(classifiers[4], testing_set) * 100) + N)
            out.write("\nVote Classifier,"
                        +str(nltk.classify.accuracy(classifiers[5], testing_set) * 100) + N)
