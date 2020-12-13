from nltk.corpus import stopwords
from NaturalLanguage.custom_NLTK_Utils import AlgoParams
from nltk.probability import FreqDist
from nltk import word_tokenize
from nltk import pos_tag
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
    return documents

def assemble_all_words(param):
    """
    Parameters
    ----------
    param : AlgoParam
        AlgoParam Object.

    Returns
    -------
    all_words : nlkt.FreqDist object
        A freqency distibution of (word, count) of every word that could be
        treated as a feature given the paramters.
        Timecost of Linear the Size of the labeled dataset

    """
    short_pos_words = word_tokenize(param.PosExamples)
    short_neg_words = word_tokenize(param.NegExamples)
    all_words = []
    for w in short_pos_words:
        all_words.append(w.lower()) 
    for w in short_neg_words:
        all_words.append(w.lower())
        
    all_words_lessLimits = limit_features(all_words, param)
    all_wordsFRQ = FreqDist(all_words_lessLimits)
    return all_wordsFRQ


def limit_features(all_words, param): # untested
    """
    Removes Stopwords and the Parts of Speech when appropriate.
    
    It might be faster to only walk through the all_words once, rather than twice.
    Some testing would clear this up. 
    
    It is very costly to remove words by type. 1 minute for 150,000 words

    Parameters
    ----------
    all_words : List
        Contains every single word in the entire dataset
    param : TYPE
        AlgoPram, accessing AlgoParam.the_stop and AlgoParam.PartsOfSpeech
    Returns
        Nothing it just reduces all_words when approprite
    -------
    """
    ans = all_words
    if param.the_stop:
        temp = []
        stop_words = set(stopwords.words('english')) # I added this to remove all the stop words
        for w in all_words:
            if(w not in stop_words):
                temp.append(w)
        all_words = temp
        ans = temp
    
    if param.PartsOfSpeech != "*":
        print('you are limiting all_words by {}'.format(param.PartsOfSpeech))
        print('Before you remove everything that is not a {} you have {} words'.format((param.PartsOfSpeech),len(all_words)))
        temp = []
        for word in all_words:
            pos = pos_tag(word_tokenize(word),tagset='universal')
            # print('type of pos_tag: {} word {}'.format(str(pos),word)) for debugging
            if(pos[0][1] == param.PartsOfSpeech):
                temp.append(pos[0][0])
                # print('type of pos_tag: {}'.format(str(pos)))
        print('After  you remove everything that is not a {} you have {} words'.format((param.PartsOfSpeech),len(temp)))
        print('OUTSIDE OF MAIN here is a sample of words in all_words: {}'.format(str(temp[:20])))
        ans = temp
    return ans
        
  
        
def assemble_word_features(all_words, param):
    """
    Parameters
    ----------
    all_words : nltk.FreqDist object
        .
    param : AlgoParam
        only using AlgoParam.NmostFrequent

    Returns
    -------
    word_features : A list of all the words to be treated as features. 
    What words, their types, and the total number of words will influence
    of the classifiers

    """
    # source: https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    dict(sorted(all_words.items(), key=lambda item: item[1]))
    word_features = list(all_words)[:param.NmostFrequent]
    return word_features
    
    
def create_feature_sets(param):
    """
    Parameters
    ----------
    all_words : nltk.FreqDist object
        .
    param : AlgoParam
        only using AlgoParam.NmostFrequent

    Returns
    -------
    
    This will create a list of tuples representing
    [Dictionary of

                 (word: boolean(Occurs In Most common N words) # this has a lot of words in it
                  "String if this review is postive or negative".
                  These are the objects that you split into Train / Test that will train the algo
    Example
    [({"great":True, "fish": False ...}, "pos"), ({"amazing":True, "kevin": False ...}, "neg")...]
    """
    documents = assemble_Documents(param) 
    all_words = assemble_all_words(param)
    word_features = assemble_word_features(all_words, param)
    feature_sets = [(find_Features(text, word_features), category) 
                    for (text, category) in documents]
    random.shuffle(feature_sets)
    return feature_sets # don't shuffle the feature_sets after this point
    










# def lookAtAccuracy(sample):
#     """
#     This is to get a more indepth understanding of the accuracy of a classifier

#     """
#     numCorrect =0
#     HighConfidenceCorrect =0
#     numHighConfidence =0
#     sample =1000
    
#     testingSetIn = open("C:/Users/parke/Documents/GitHub/NaturalLanguage/NaturalLanguage/OwnPrograms/pickled_TestingData.pickle", "rb")
#     TestingSet = pickle.load(testingSetIn)
#     testingSetIn.close()
    
#     classifierList = open("C:/Users/parke/Documents/GitHub/NaturalLanguage/NaturalLanguage/OwnPrograms/pickled_TrainedClassifierList.pickle", "rb")
#     classifiers = pickle.load(classifierList)
#     classifierList.close()
#     voted_classifier = VoteClassifier(classifiers[0],classifiers[1],
#                                           classifiers[2],classifiers[3],
#                                           classifiers[4])

    
#     for i in range(sample):
#     # print("Classification: ", voted_classifier.classify(TestingSet[i][0]),
#     #       "Correct:",TestingSet[i][1],
#     #       "Confidence %: ", voted_classifier.confidence(TestingSet[i][0])*100 
#     #       )
#         print(i)
#         if TestingSet[i][1] == voted_classifier.classify(TestingSet[i][0]):
#            numCorrect= numCorrect+1
           
#         if TestingSet[i][1] == voted_classifier.classify(TestingSet[i][0]):
#            if voted_classifier.confidence(TestingSet[i][0]) == 1:
#                 HighConfidenceCorrect = HighConfidenceCorrect+1
                
#         if voted_classifier.confidence(TestingSet[i][0]) ==1:
#             numHighConfidence= numHighConfidence+1
#     print(numCorrect/sample)
#     print(HighConfidenceCorrect/sample)
#     print(HighConfidenceCorrect/numHighConfidence)
