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
    shortPos = open("short_reviews/shortPositive.txt","r").read()
    shortNeg = open("short_reviews/shortNegative.txt","r").read()
    paramList = []
    for stopWords in (True,False):
        for N in (2000,3000,4000,5000):
                paramList.append(AlgoParams.AlgoParams(stopWords, N, shortPos, shortNeg, ["*"]))
    return paramList

def create_Feature_sets_list(param):
    all_words = dl.assemble_all_wordsFRQDIST(param)
    feature_sets = dl.create_feature_sets(param)
    return feature_sets

    
def getTestData(feature_sets):
    N = int(len(feature_sets)*.9) #90% in training data BROKEN
    TestingData = feature_sets[N:]
    return TestingData

def getTrainData(feature_sets):
    N = int(len(feature_sets)*.9) #90% in training data BROKEN
    TrainingData = feature_sets[:N]
    return TrainingData

def CreateAndTrain_Classifiers(TrainingData):
    TrainingData = getTrainData(TrainingData)
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

def writeAlgoEvaluation(param, classifiers, results):
    TestingSet = getTestData(FS)
    with open("AlgoEvalutationResults.txt","a+") as out:
        out.write("\n----------------------------------------------\n")
        
        res = ("StopWords       : " + str(param.the_stop) + 
               "\nN Most Frequent : " + str(param.NmostFrequent) +
               "\nPartsOfSpeech   : " + str(param.PartsOfSpeech) +"\n"
                )
        out.write(res)
        out.write("\nAccuracy of Naive Bayes                   :"+str(nltk.classify.accuracy(classifiers[0], TestingSet)*100))
        out.write("\nAccuracy of SGD Classifiers               :"+str(nltk.classify.accuracy(classifiers[1], TestingSet)*100))
        out.write("\nAccuracy of Bernoulli Naive Bayes         :"+str(nltk.classify.accuracy(classifiers[2], TestingSet)*100))
        out.write("\nAccuracy of Linear Support Vector Machine :"+str(nltk.classify.accuracy(classifiers[3], TestingSet)*100))
        out.write("\nAccuracy of Logistic Regression           :"+str(nltk.classify.accuracy(classifiers[4], TestingSet)*100))
        out.write("\nAccuracy of Vote Classifier               :"+str(nltk.classify.accuracy(classifiers[5], TestingSet)*100))
        localRes = (param, nltk.classify.accuracy(classifiers[5], TestingSet)*100)
        print("localres: " + localRes)
        results.append(localRes)
        out.write("\n----------------------------------------------\n\n")

start = datetime.datetime.now()
print('you have started')


                    ### You broke this and need to restore it from the commits you made yesterday. 
                    ### I don't yet know how to do this


paramList = createParamList()
counter =1
for p in paramList:
    start2 =  datetime.datetime.now()
    print("You are at this call of 8:",end="")
    print(counter)
    counter = counter +1
    FS = create_Feature_sets_list(p)
    results =[()]
    classifiers = CreateAndTrain_Classifiers(FS)
    writeAlgoEvaluation(p,classifiers,results)
    # the parts of speech stuff is broken, otherwise this is still good
    print("A pass of the for loop took this long: ", end="")
    print(datetime.datetime.now()-start2)
    
# results =[()]
# FS = create_Feature_sets_list(paramList[0]) # Part of the random is wrong here
# classifiers = CreateAndTrain_Classifiers(FS)
# writeAlgoEvaluation(paramList[0],classifiers,results)

print('finished')
print(datetime.datetime.now() -start)