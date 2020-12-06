from nltk.corpus import stopwords
import nltk
import random

def find_Features(document,word_features):
    """
        I have no idea what this method does. I need to rewatch the lecture series on it
        
        word Featuers is the list of tuples? that are most frequent in each of the different 
        positive or negative sentimet texts
        
        Later you pass this a tweet and it will parse out the features from it.
        and be passed into the Voted Classifer
        
    """
    limit_features(document) # this removes stop words from consideration
    words = set(document) 
    features = {}
    for w in word_features:
        features[w] = (w in words) # this a boolean
    return features  


def assemble_Documents(PositiveExamples, NegativeExamples):
    documents = [] # document is a tuple of (review, classifcation)
    for r in PositiveExamples.split('\n'):
        documents.append((r,"pos"))
    for r in NegativeExamples.split('\n'):
        documents.append((r,"neg"))
    return documents

def assemble_all_wordsFRQDIST(PositiveExamples, NegativeExamples):
    short_pos_words = nltk.word_tokenize(PositiveExamples)
    short_neg_words = nltk.word_tokenize(NegativeExamples)
    
    all_words = []
    for w in short_pos_words:
        all_words.append(w.lower()) 
    for w in short_neg_words:
        all_words.append(w.lower())
        
    limit_features(all_words)
    all_words = nltk.FreqDist(all_words)
    return all_words

    
    
def limit_features(all_words):
    """
    Parameters
    ----------
    all_words : TYPE
        I am not sure what type this is it might be a tuple like ("word", 'pos')
        or it might be a list of every word
        Later you can chnage this method to parse by part of speech
        
        That will make the algos train better. 
    """
    stop_words = set(stopwords.words('english')) # I added this to remove all the stop words
    all_words = [w for w in all_words if (not w in stop_words)] 
    
def assemble_word_features(all_words, numFeaturesToConsider):
    return (all_words.keys())[:numFeaturesToConsider]

    
def create_feature_sets(PositiveExamples, NegativeExamples):
    documents = assemble_Documents(PositiveExamples, NegativeExamples)
    all_words = assemble_all_wordsFRQDIST(PositiveExamples,NegativeExamples)
    word_features = assemble_word_features(all_words, 3000)
    
    
    feature_sets = [(find_Features(text, word_features), category) 
                    for (text, category) in documents]
    
    return random.shuffle(feature_sets)


def lookAtAccuracy(sample):
    numCorrect =0
    HighConfidenceCorrect =0
    numHighConfidence =0
    sample =1000
    
    testingSetIn = open("C:/Users/parke/Documents/GitHub/NaturalLanguage/NaturalLanguage/OwnPrograms/pickled_TestingData.pickle", "rb")
    TestingSet = pickle.load(testingSetIn)
    testingSetIn.close()
    
    classifierList = open("C:/Users/parke/Documents/GitHub/NaturalLanguage/NaturalLanguage/OwnPrograms/pickled_TrainedClassifierList.pickle", "rb")
    classifiers = pickle.load(classifierList)
    classifierList.close()
    voted_classifier = VoteClassifier(classifiers[0],classifiers[1],
                                          classifiers[2],classifiers[3],
                                          classifiers[4])

    
    for i in range(sample):
    # print("Classification: ", voted_classifier.classify(TestingSet[i][0]),
    #       "Correct:",TestingSet[i][1],
    #       "Confidence %: ", voted_classifier.confidence(TestingSet[i][0])*100 
    #       )
        print(i)
        if TestingSet[i][1] == voted_classifier.classify(TestingSet[i][0]):
           numCorrect= numCorrect+1
           
        if TestingSet[i][1] == voted_classifier.classify(TestingSet[i][0]):
           if voted_classifier.confidence(TestingSet[i][0]) == 1:
                HighConfidenceCorrect = HighConfidenceCorrect+1
                
        if voted_classifier.confidence(TestingSet[i][0]) ==1:
            numHighConfidence= numHighConfidence+1
    print(numCorrect/sample)
    print(HighConfidenceCorrect/sample)
    print(HighConfidenceCorrect/numHighConfidence)
