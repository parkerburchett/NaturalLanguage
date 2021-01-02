"""
This a series of methods that will take in a string,
lable,
remove by part of speech,
lemmatize it,

returns a list of words

It is unclear if you can train a Naive Bayes on boolean Vector: boolean vector pairs. Of if you needs some other format


"""

from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

my_lemmatizer = WordNetLemmatizer() # it might be faster to initalize this once rather than at every call to this method


def tag_speech(text):
    """
    Parameters:
        text: a string
    Returns:
        tagged: list of (word, part of speech) tuples
    """
    tok = word_tokenize(text)
    tagged = pos_tag(tok, tagset='universal') # this is smart enough to realize context influences part of speech

    return tagged


def limit_parts_of_speech(tagged, default_parts_of_speech=True, custom_parts_of_speech=None):
    """
    Parameters:
        tagged: list of (word, part of speech) tuples
        source: https://universaldependencies.org/u/pos/
        default_parts_of_speech: Boolean, use the default or not.
        The default is No
                     determiners,
                     punctuation,
                     symbols,
                     pronouns,
                     proper nouns
                     coordinating conjunctions,
                     numerals,

        custom_parts_of_speech: a list of parts of speech, using the universal tagset.
                                Include the tags for Parts of speech to exclude.
                                If you don't want to exclude anything call this as
                                default_parts_of_speech=False, custom_parts_of_speech=[]

    Returns:
        limited_words: word, pos tuple if not excluded
    """

    if default_parts_of_speech:
        # this is the default. I don't have a good warrant for this list. It is intuition.
        to_exclude = ['DET','PUNCT','SYMB','PRON','PROPN','CCONJ','NUM']
    else:
        to_exclude = custom_parts_of_speech
    # numpy might make this faster depending on the speed of this process
    limited_words = []
    for word in tagged:
        # you might want to change this to list comprehension to make it faster

        #limited_words = [w[0] for w in words if w[0] not in to_exclude # untested

        if word[1] not in to_exclude:
            limited_words.append(word)

    return limited_words

def convert_to_lemma(words):
    """
        Parameters:
            words: a list of strings of where each string is a word that is not used as an excluded part of speech
        Returns:
            lemmas: A list of words that are lemmas of words

        Note: this uses the WordNetLemmatizer() and is only instantiated once for speed.
    """

    lemmas =[]

    # there is a problem with the universal tagset to make the pos match the format for the lemmatizer.
    # someone encountered this problem before.
    # https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
    for w in words:
        lemmas.append(my_lemmatizer.lemmatize(w[0],pos=w[1]))
    return lemmas



t = tag_speech("I need help from the governement to the the permitting in order for my permits for my permit collection")
#[('    They', 'PRON'), ('refuse', 'VERB'), ('to', 'PRT'), ('permit', 'VERB'), ('us', 'PRON'), ('to', 'PRT'),
        # ('obtain', 'VERB'), ('the', 'DET'), ('refuse', 'NOUN'), ('permit', 'NOUN')]


words = limit_parts_of_speech(t)

lemmas = convert_to_lemma(words)
print('fin')