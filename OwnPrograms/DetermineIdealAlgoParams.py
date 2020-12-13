"""
Explanation of the goals of this program

When you create each classification algorithm there are several arbritray paramaters 
that you decided on to start. The purpose of this file, and the method that go along with it.
are to decided what version of those params are the most accurate. 


When you look at the data labeling paramaters the params that go into the classifiers

These are the parts

    (
    PositveExamples: A String[] that is read in from the file containing every positive reivew
    NegativeExamples: A String[] read in from the negative examples
    
    
    documents : a list of tuples containging An element of (PositiveExamples, "Pos / Neg")
                             PostiveExamples[i] is a string here.
    
    all_words[] : a frequency distribution of how often each word occurs in the entire sample
        You can do things to limit what types of words are considerd here.
                limit_features() serves this purpose
                
    word_features: the N most common words, where you can change N. In the entire sample
    
    feature_sets: A list of Dictionary, String tuples 
    representing the boolean of if a word is in the common features and if the review is postiive or negative 
     
    N mostCommon Words
    
    
    Regex of what Parts of speech to consider.
    
    Boolean to consider stopwords.
     )

Use the data object


from  custom_NLTK_Utils.README.txt

For Each , With_stopWords and WithoutStopWOrds: (2)
    For All combinations of (Adjectives, Adverbs, Verbs): (6)
        For each size of word_features 2000, 3000, 4000 (3):
            Generate a new FeatureSet based on the above Params
            Train the 5 Algos
            Train the VoteClassifer
            Write to a File the accuracy of the 5 methods 
            Write to a File the accuracy of the VoteClassifer
            Write to a file lookAtAccuracy() for the Vote CLassifier
            Save as a tuple (RegexOFPosToConsider, SizeOf WordFeatures, 
            Accuracy of VoteClassifer, ALL,
            Vote Clasifer Accuracy at 60% confident, Accuracy at 80%, Accuracy at 100% )
            Num testing set at 60% confident, num at 80% confident and num at 100% confident
                Write that tuple to a different


For whatever reason, this is a very bad tester. It is just about 50% accurate

"""


from NaturalLanguage.custom_NLTK_Utils import AlgoParams
from NaturalLanguage.custom_NLTK_Utils import dataLabeling as dl
from NaturalLanguage.custom_NLTK_Utils import VoteClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier 
from nltk.classify import ClassifierI
from statistics import mode
from sklearn.svm import SVC, LinearSVC, NuSVC
import nltk
import datetime
import random

def createParamList():
    shortPos = open("C:/Users/parke/Documents/GitHub/NaturalLanguage/NaturalLanguage/OwnPrograms/short_reviews/shortPositive.txt","r").read()
    shortNeg = open("C:/Users/parke/Documents/GitHub/NaturalLanguage/NaturalLanguage/OwnPrograms/short_reviews/shortNegative.txt","r").read()
    paramList = []
    for f in ("NOUN", "ADJ", "VERB","ADV", "*"):
        paramList.append(AlgoParams.AlgoParams(False, 1000, shortPos, shortNeg, f))
    return paramList
    
def getTestData(feature_sets):
    N = int(len(feature_sets)*.9) 
    TestingData = feature_sets[N:]
    return TestingData

def getTrainData(feature_sets):
    N = int(len(feature_sets)*.9) 
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
    # change where this writes to so it writes to the current diricetory
    with open("PartsOfSpeech2_AlgoEvalutationResults.csv","a+") as out:
        out.write("\n----------------------------------------------\n")
        ParamDetails = ("Remove StopWords: " + str(param.the_stop) + 
                       "\nN Most Frequent : " + str(param.NmostFrequent) +
                       "\nPartsOfSpeech   : " + str(param.PartsOfSpeech)
                        )
        out.write(ParamDetails)
        out.write("\nNaive Bayes Most informativeFeatures: {}".format(classifiers[0].most_informative_features(10)))
        out.write("\nAccuracy of Naive Bayes                   :"+str(nltk.classify.accuracy(classifiers[0], TestingSet)*100))
        out.write("\nAccuracy of SGD Classifiers               :"+str(nltk.classify.accuracy(classifiers[1], TestingSet)*100))
        out.write("\nAccuracy of Bernoulli Naive Bayes         :"+str(nltk.classify.accuracy(classifiers[2], TestingSet)*100))
        out.write("\nAccuracy of Linear Support Vector Machine :"+str(nltk.classify.accuracy(classifiers[3], TestingSet)*100))
        out.write("\nAccuracy of Logistic Regression           :"+str(nltk.classify.accuracy(classifiers[4], TestingSet)*100))
        out.write("\nAccuracy of Vote Classifier               :"+str(nltk.classify.accuracy(classifiers[5], TestingSet)*100))
        out.write("\n----------------------------------------------\n\n")

start = datetime.datetime.now()
print('you have started')
paramList = createParamList()
counter =1
for p in paramList:
    try:
        start2 =  datetime.datetime.now()
        print("At {} of {} where N={}".format(counter,len(paramList),p.NmostFrequent))
        counter = counter +1
        FS = dl.create_feature_sets(p)
        print('Created Feature_sets {}:'.format((datetime.datetime.now()-start2)))
        classifiers = CreateAndTrain_Classifiers(FS)
        print('Trained Classifiers: {}:'.format((datetime.datetime.now()-start2)))
        writeAlgoEvaluation(p,classifiers,FS)
        print('Tested Classifiers: {}:'.format((datetime.datetime.now()-start2)))
    
    except (ValueError):
        print('you had an error here')
        print(ValueError)
    


print('finished')
print(datetime.datetime.now() -start)