from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
import numpy as np
import string
import random

from NaturalLanguage.custom_NLTK_Utils.CustomLemmatizer import CustomLemmatizer

def find_features_lemma(list_of_words, word_features):
    """
    Parameters:
        list_of_words: a list of the lemmatized words found in a tweet.
        word_features: the list of words to be treated as features in the model

    returns:
        features: A dictionary of word: Boolean pairs.
            Represents the presence of absences of a word in a tweet that is also in the word_features.
    """
    features = {}
    print('in find_Features')
    for w in word_features:
        features[w] = (w in list_of_words)  # this a boolean
    return features

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
    print('at start of find_Features')

    # for an unknown reason it break when you call word_tokenize(document)
    print(document)

    # document i think is already a list of words. you are breaking it by calling word_tokenize on it
    words = word_tokenize(document)
    print('evidence at broke at word_tokenize(document)')
    features = {}  # empty dictionary
    print('in find_Features')
    for w in word_features:
        features[w] = (w in words)  # this a boolean
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
            documents.append((review, "Positive"))
        for review in param.NegExamples.split('\n'):
            documents.append((review, "Negative"))

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

    if param.PartsOfSpeech != "*":  # this or reduce to only NOUNS or only VERBS but not to NOUNS and VERBS
        temp = []
        for word in all_words:
            pos = pos_tag(word_tokenize(word), tagset='universal')
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
        with open("word_featuresList.txt", "a+") as out:
            out.write(param.PartsOfSpeech)
            for w in range(min(100, len(word_features) - 1)):
                out.write(word_features[w] + ", ")
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
    documents = assemble_Documents(param, randomized=True)
    all_words = assemble_all_words(param)
    word_features = assemble_word_features(all_words, param)
    feature_sets = [(find_Features(text, word_features), category)
                    for (text, category) in documents]
    random.shuffle(feature_sets)
    return feature_sets  # don't shuffle the feature_sets after this point


def assemble_kaggle_documents(inputFile):
    """
    This is for converting a .csv file into the same format as the rest of the module is using.
    and 0 = negative  4= positive

    Parameters: inputFile a .csv file in the form of (4, "tweet with good sentiment") or (0, "tweet with bad sentiment")

    Returns:
        a list of Tuples representing (Tweet, category) for every labeled review.
    """
    lines = inputFile.readlines()
    documents = []
    for line in lines:
        splitLine = line.split(',', 1)
        tweet = str(splitLine[1]).lower()
        # it is unclear if you need to word tokenize the document here
        if int(splitLine[0]) == 0:
            category = "Negative"
        elif int(splitLine[0]) == 4:
            category = "Positive"

        LabeledReview = (tweet, category)
        documents.append(LabeledReview)
    random.shuffle(documents)
    inputFile.close()
    return documents


def create_my_stop_words():
    """
        Param: None
        Returns:
            A list of words to be considered stopwords
    """
    my_stop_words = stopwords.words()
    puncts = string.punctuation
    # add punctuation to stopwords.
    # you might want to remove ! since it has a positive association
    for p in puncts:
        my_stop_words.append(p)
    return my_stop_words


def kaggle_assemble_word_features(documents, num_features, remove_stopwords=False):
    """
        Parameters:
            documents:  documents[0]: a tweet as plain text.
                        documents[1]: the label of the tweet.

            num_features: an int representing how many unique words to treat as features.
                        this sorted

            remove_stopwords: boolean for removing stopwords

        Returns:
            a list of words to be treated as features in it orthagonal order

    """
    if remove_stopwords:
        my_stop_words = create_my_stop_words()
    else:
        my_stop_words = []  # empty list

    all_words = []
    for d in documents:
        words = word_tokenize(d[0])
        for w in words:
            if w not in my_stop_words:
                all_words.append(w.lower())

    all_words_frq = FreqDist(all_words)
    word_features_as_tuples = all_words_frq.most_common(num_features)  # this is so clean
    word_features = []
    for w in word_features_as_tuples:
        word_features.append(w[0])

    sorted(word_features)
    # sorted is the orthogonal ordering.
    # it is very important that the word_features does not have a different anytime you would use it
    # you call sorted(word_features)
    # this has a very small time cost O(N * log(N) where N = num_features.
    return word_features


def text_to_vector(text, word_features):  # untested
    """
        Description:
            Map some text onto a boolean vector based on word_features. Uses 'Bag of Words" approach.
            I think of this as a encoding a bag of words based on the word_features
        Parameters:
            text: a string to be converted
            word_features: the words to be treated as vectors.
        Returns:
            vector: a np boolean vector based on the contents and orientation of _word_features
    """

    words = set(word_tokenize(text))  # you only care if a word occurs at least once
    vector = np.zeros(len(word_features), dtype=bool)
    for i in range(len(word_features)):
        if word_features[i] in words:
            vector[i] = True
    return vector


def vector_to_words(vector, word_features):  # untested
    """
    Parameters:
        vector: boolean vector of some text.
        word_features: a list of words to be treated as features
    Returns:
        words: a list of words that is represented by the vector. Ignores duplicates
        I think of this method as decoding a vector based on the word_features.
    """
    if len(vector) != len(word_features):
        raise ValueError('You are trying to decode a vector based on the wrong word_features'
                         '\nvector len {}, word_features len {}'.format(len(vector), len(word_features)))

    words = []
    for value in range(len(vector)):  # untested
        if vector[value]:
            words.append(word_features[value])

    return words


def assemble_lemma_documents(input_file, short=True):
    """
    Convert the labeled tweets into
    lemmas: category tuples.

    This should be significantly faster since it uses list comprehension
    """
    lines = input_file.readlines()
    split_lines =[]
    for line in lines:
        split_lines.append(line.split(',', 1))

    my_lemmatizer = CustomLemmatizer()
    print('before creating documents')


    num_lines = len(split_lines)
    documents = []
    if short == False:
        for i in range(100000): # untested
            lemmas = my_lemmatizer.determine_lemmas(split_lines[i][1])
            category = numeral_to_category(split_lines[i][0])
            a_doc = np.array((lemmas, category))
            documents.append(a_doc)
            if i % 1000 == 0:
                print(i)

    else:
        documents = [(my_lemmatizer.determine_lemmas(line[1]),
                    numeral_to_category(line[0]))
                     for line
                     in split_lines[:1000]]
    print('after creating documents')
    random.shuffle(documents)
    docs_as_np =np.array(documents)

    return docs_as_np


def assemble_lemma_word_features(documents, num_features=2000):
    """
    this assemble word_feature based on the lemmatized version of a tweet.
    """


    all_words =[]
    for d in documents:
        all_words.extend(d[0])

    # all words here should only be lemmas
    all_words_frq = FreqDist(all_words)
    word_features_as_tuples = all_words_frq.most_common(num_features)  # this is so clean
    word_features = [w[0] for w in word_features_as_tuples]

    sorted(word_features) # might not do anything

    return word_features


def numeral_to_category(numeral):
    if int(numeral) == 0:
        return False
    else:
        return True
