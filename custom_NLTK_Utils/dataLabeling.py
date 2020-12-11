from nltk.corpus import stopwords
from NaturalLanguage.custom_NLTK_Utils import AlgoParams
from nltk.probability import FreqDist
from nltk import word_tokenize
import random

def find_Features(document, word_features):
    """
        source: https://www.youtube.com/watch?v=-vVskDsHcVc&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL&index=12
        
        This method takes a string of Text, "documents" 
        It removes all duplicate words, then it creates a dictionary object
        where The key is each unique word and the value is a boolean.
        The boolean represented if that word occurs in the most common N words
        
        Features is then a  dictionary that will look like this {"great": True, "fish": False}
        This would be if Great is the word_features, and fish is not.
        
        There are many words that are just so rare that they don't' have predictive value. 
        those are assigned false.

    """
    # limit_features(document, ifStop, PartsOfSpeech) # this removes stop words from consideration
    words = word_tokenize(document) # this was where the bug was in DetermineIdealALgoParams.py was It was words = set(document)
    features = {} # empty dictionary
    for w in word_features:
        features[w] = (w in words) # this a boolean
    return features  


def assemble_Documents(param):
    documents = [] # document is a tuple of (review, classifcation)
    for r in param.PosExamples.split('\n'):
        documents.append((r,"Positive"))
    for r in param.NegExamples.split('\n'):
        documents.append((r,"Negative"))
    random.shuffle(documents)
    return documents

def assemble_all_wordsFRQDIST(param):
    short_pos_words = word_tokenize(param.PosExamples)
    short_neg_words = word_tokenize(param.NegExamples)
    all_words = []
    for w in short_pos_words:
        all_words.append(w.lower()) 
    for w in short_neg_words:
        all_words.append(w.lower())
        
    limit_features(all_words, param)
    all_words = FreqDist(all_words)
    return all_words

def FromLecture_assemble_all_words_FREQDIST(param):
    """
    Source:
    https://www.youtube.com/watch?v=eObouMO2qSE&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL&index=19
    This makes it so that allwords is limited by both stop words and parts of speech
    """
    all_words = []
    Examples = [param.PosExamples, param.NegExamples]
    for e in Examples:
        for review in e.split('\n'):
            words = nltk.word_tokenize(review)
            pos_inthisReveiw = nltk.pos_tag(words)
            for w in pos_inthisReveiw:
                if (w[1][0]) in param.PartsOfSpeech:
                    all_words.append(w[0].lower())
    limit_features(all_words, param)
    all_words = nltk.FreqDist(all_words)
    return all_words
            
    
def limit_features(all_words, param):
    """
        Need to add way to parse REGEX from PartsOfspeech 
        That will make the algos train better. 
    """
    if(param.the_stop):
        stop_words = set(stopwords.words('english')) # I added this to remove all the stop words
        all_words = [w for w in all_words if (not w in stop_words)]
    
    
def assemble_word_features(all_words, param):
    """
    returns list of the frequent N unique words
    """
    
    dict(sorted(all_words.items(), key=lambda item: item[1]))

    word_features = list(all_words)[:param.NmostFrequent]
    return word_features
    
def create_feature_sets(param):
    """
    This will create a list of tuples representing
    [Dictionary of

                 (word: boolean(Occurs In Most common N words) # this has a lot of words in it
                  "String if this review is postive or negative".
                  These are the objects that you split into Train / Test that will train the algo
    Example
    [({"great":True, "fish": False ...}, "pos"), ({"amazing":True, "kevin": False ...}, "neg")...]
    """
    documents = assemble_Documents(param) 
    all_words = assemble_all_wordsFRQDIST(param)
    word_features = assemble_word_features(all_words, param)
    # this is the key line
    feature_sets = [(find_Features(text, word_features), category) 
                    for (text, category) in documents]

    random.shuffle(feature_sets)
    return feature_sets # don't shuffle the feature_sets after this point
    


def lookAtAccuracy(sample):
    """
    This is to get a more indepth understanding of the accuracy of a classifier

    """
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
