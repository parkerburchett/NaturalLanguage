from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk import word_tokenize
from nltk import pos_tag
import random

def find_Features(document, word_features):
    """
    Parameters
    ----------
    document : A single document that needs to be categorized.
    word_features : a list of every word that will be treated as a feature.
        source: https://www.youtube.com/watch?v=-vVskDsHcVc&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL&index=12
    Returns
    -------
    features : A dictionary that is the length of the word_features where the key is the word
    and the value is a boolean representing if that word is present in document.
    """
    words = word_tokenize(document)
    features = {} # empty dictionary
    for w in word_features:
        features[w] = (w in words) # this a boolean
    return features  


def assemble_Documents(param, randomized=False):
    """
    Parameters: Param: AlgoParam Object using AlgoParam.PosExamples and AlgoParam.NegExamples
    randomized:Optional Default False.
    If true this will randomly label documents
    randomized=True is for a proof of concept that the shows there needs to be an underlying relationship between
    A input and output. If there is not an output, theory says accuracy should always be close to 50%
    Returns:
        a list of Tuples representing (review, category) for every labeled review.
    """
    documents = []  # document is a tuple of (review, category)
    examples = (param.PosExamples.split("\n"), param.NegExamples.split("\n"))
    if randomized:
        for e in examples:
            for r in e:
                if bool(random.getrandbits(1)):
                    documents.append((r, "Positive"))
                else:
                    documents.append((r, "Negative"))
    else:
        for review in param.PosExamples.split('\n'):
            documents.append((review,"Positive"))
        for review in param.NegExamples.split('\n'):
            documents.append((review,"Negative"))

    return documents


def assemble_all_words(param):
    """
    Parameters
    ----------
    param : AlgoParam
       Accesses AlgoParam.PosExamples and AlgoParam.NegExamples
    Returns
    -------
    all_words : nlkt.FreqDist object
       A frequency distribution of (word, count) of every word that could be
       treated as a feature given the parameters.
       Time cost of linear of the number of words
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


def limit_features(all_words, param):
    """
    Removes stop words and or limits to a single part of speech. Time cost is linear of all_words
    Parameters
    ----------
    all_words : List
        Contains every single word in the entire dataset.
    param : TYPE
        AlgoPram, accessing AlgoParam.the_stop and AlgoParam.PartsOfSpeech
    Returns
        A new list of all_words that less what was removed
    -------
    """
    ans = all_words
    if param.the_stop:
        temp = []
        stop_words = set(stopwords.words('english'))
        for w in all_words:
            if w not in stop_words:
                temp.append(w)
        all_words = temp
        ans = temp
    
    if param.PartsOfSpeech != "*": # this or reduce to only NOUNS or only VERBS but not to NOUNS and VERBS
        temp = []
        for word in all_words:
            pos = pos_tag(word_tokenize(word),tagset='universal')
            if pos[0][1] == param.PartsOfSpeech:
                temp.append(pos[0][0])
        ans = temp
    return ans
        

def assemble_word_features(all_words, param, write=False):
    """
    Parameters
    ----------
    all_words : nltk.FreqDist object
    param : AlgoParam
        Accessing AlgoParam.NmostFrequent
    Returns
    -------
    word_features : A list of all the words to be treated as features
    """
    # source: https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value

    dict(sorted(all_words.items(), key=lambda item: item[1]))
    try:
        word_features = list(all_words)[:param.NmostFrequent]
    except:
        word_features = list(all_words)
        # this only triggers when there are not enough Unique Words in all_words
        # This is the case for example when you are limiting by punctuation
        # and there are not enough unique punctuation symbols
        # it is currently untested. I need to verify that it works
    if write:
        with open("word_featuresList.txt","a+") as out:
            out.write(param.PartsOfSpeech)
            for w in range(min(100, len(word_features)-1)):
                out.write(word_features[w]+", ")
            out.write("\n")
    return word_features
    
    
def create_feature_sets(param):
    """
    This stitches all the sub methods together.
    Parameters
    ----------
    param : AlgoParam
    Returns
    feature_sets : A list of tuples of ({WordFeatures[i]: Boolean is in this document}, category) ...
    You can think of the word_features dictionary as a vector of len(word_features) dimensions that is used
    to train the different algorithms where to get relationships between (VECTOR, Cateogory).
    -------
    Example
    [({"great":True, "kevin": False ...}, "Positive"), ({"great":False, "kevin": False ...}, "Negative")...]
    """
    documents = assemble_Documents(param)
    all_words = assemble_all_words(param)
    word_features = assemble_word_features(all_words, param)
    feature_sets = [(find_Features(text, word_features), category) 
                    for (text, category) in documents]
    random.shuffle(feature_sets)
    return feature_sets # don't shuffle the feature_sets after this point