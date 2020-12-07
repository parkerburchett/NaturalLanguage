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
        for N in (1000,2000,3000):
            for PartOfSpeech in (["J","R","V"],
                                 ["J","R"],["J","V"],["R","V"],
                                 ["J"],["R"],["V"]):
                paramList.append(AlgoParams.AlgoParams(stopWords, N, shortPos, shortNeg, PartOfSpeech))
    return paramList

def create_Feature_sets_list(param):
    documents = dl.assemble_Documents(param.PosExamples, param.NegExamples)
    all_words = dl.assemble_all_wordsFRQDIST(param.PosExamples, param.NegExamples, 
                                             param.the_stop, param.PartsOfSpeech)
    word_features = dl.assemble_word_features(all_words, param.NmostFrequent)
    feature_sets = dl.create_feature_sets(documents, word_features, param.the_stop, param.PartsOfSpeech)
    random.shuffle(feature_sets) # don't shuffle it after this
    return feature_sets
    
def getTestData(FS):
    N = int(len(fs)*.9) #90% in training data
    TestingData = feature_sets[N:]
    return TestingData

def getTrainData(FS):
    N = int(len(fs)*.9) #90% in training data
    TrainingData = feature_sets[:N]
    return TrainingData
    

def CreateAndTrain_Classifiers(FS):
    N = int(len(fs)*.9) #90% in training data
    TrainingData = getTrainData(FS)
    TestingData = getTestData(FS)
    
    TrainedClassifierList = []
    NBClassifer = nltk.NaiveBayesClassifier.train(TrainingData)
    TrainedClassifierList.append(NBClassifer)
    print("Trained NaiveBayes", end="")
    print(datetime.datetime.now()-start)
    
    c = SklearnClassifier(SGDClassifier())
    c.train(TrainingData)
    print("Trained SGD Classifier ", end="")
    TrainedClassifierList.append(c)
    print(datetime.datetime.now() -start)
    
    c = SklearnClassifier(BernoulliNB())
    c.train(TrainingData)
    print("Trained Bernoulli Naive Bayes ", end="")
    TrainedClassifierList.append(c)
    print(datetime.datetime.now() -start)
    
    c = SklearnClassifier(LinearSVC())
    c.train(TrainingData)
    print("Trained Linear Support Vector Machine ", end="")
    TrainedClassifierList.append(c)
    print(datetime.datetime.now() -start)
    
    c = SklearnClassifier(LogisticRegression())
    c.train(TrainingData)
    print("Trained Logistic Regression ", end="")
    TrainedClassifierList.append(c)
    print(datetime.datetime.now() -start)
    
    paramList = createParamList();
    print("It took this long to Train 5 Classifiers:", end="")
    print(datetime.datetime.now() -start)
    
    voted_classifier = VoteClassifier.VoteClassifier(classifiers[0],
                                                     classifiers[1],
                                                     classifiers[2],
                                                     classifiers[3],
                                                     classifiers[4])
    classifiers.append(voted_classifier)
    return classifiers

def writeAlgoEvaluation(param, classifiers, results):
    TestingSet = getTestData(FS)
    with open("AlgoEvalutationResults.txt","a+") as out:
        out.write("----------------------------------------------")
        out.write(toString(param))
        out.write("Accuracy of Naive Bayes:", (nltk.classify.accuracy(classifiers[0], TestingSet)*100))
        out.write("Accuracy of SGD Classifiers:", (nltk.classify.accuracy(classifiers[1], TestingSet)*100))
        out.write("Accuracy of Bernoulli Naive Bayes:", (nltk.classify.accuracy(classifiers[2], TestingSet)*100))
        out.write("Accuracy of Linear Support Vector Machine", (nltk.classify.accuracy(classifiers[3], TestingSet)*100))
        out.write("Accuracy of Logistic Regression:", (nltk.classify.accuracy(classifiers[4], TestingSet)*100))
        out.write("Accuracy of Vote Classifier:", (nltk.classify.accuracy(classifiers[5], TestingSet)*100))
        localRes = (param, nltk.classify.accuracy(classifiers[5], TestingSet)*100)
        results.append(localRes)
        print("You just wrote out the results of a param test")
        out.write("----------------------------------------------\n\n")

start = datetime.datetime.now()
print('you have started')

paramList = createParamList()
for p in paramList:
    FS = create_Feature_sets_list(p)
    ## FS is a Feature set and P is a 
    results =[()]
    classifiers = CreateAndTrain_Classifiers(FS)
    writeAlgoEvaluation(p,classifiers,results)
    print(datetime.datetime.now()-start)




#next steps pass feature sets, and the params to the Create and traing Classifier list. 

print('finished')
print(datetime.datetime.now() -start)