from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class CustomLemmatizer():
    """
        Class that keeps the lemmas into its own object for efficiency.
        use determine_lemmas(text) to get a list of lemmas of a String.

        This is very fast. Before rewriting as list comprehension:

        determine_lemmas(20 word string) * 100 times takes 2 seconds.
        determine_lemmas(20 word string) * 1,000 times takes 7 seconds.
        determine_lemmas(20 word string) * 10,000 times takes 63 seconds.

        When you deploy this for a naive bayes you need to rewrite the things as list comphension.
        
    """

    def __init__(self):
        self._lemmatizer = WordNetLemmatizer()
        # you only want to fetch the wordnet lemmatizer once. This will make it more efficient.

    def _tag_speech(self, text):
        """
        Parameters:
            text: a string
        Returns:
            tagged: list of (word, part of speech) tuples
        """
        tok = word_tokenize(text)
        tagged = pos_tag(tok, tagset='universal')  # this is smart enough to realize context influences part of speech
        return tagged  # works

    def _limit_parts_of_speech(self, tagged):
        """
        Parameters:
            tagged: list of (word, part of speech) tuples
            The default is No
                         determiners,
                         punctuation,
                         symbols,
                         pronouns,
                         proper nouns
                         coordinating conjunctions,
                         numerals
        Returns:
            limited_words: word, pos tuple if not excluded
        """
        # this is the default. I don't have a good warrant for this list. It is intuition.
        to_exclude = ['DET', 'PUNCT', 'SYMB', 'PRON', 'PROPN', 'CCONJ', 'NUM']

        # numpy might make this faster depending on the speed of this process
        limited_words = []
        for word in tagged:
            if (word[1] not in to_exclude) and (word[0] not in stopwords.words('english')):
                limited_words.append(word)

        return limited_words

    def _convert_to_lemma(self, words):
        """
            Parameters:
                words: a list of strings of where each string is a word that is not used as an excluded part of speech
            Returns:
                lemmas: A list of words that are lemmas of the contents of words

            Note: this uses the WordNetLemmatizer() and is only instantiated once for speed.
        """
        # there is a problem with the universal tagset to make the pos match the format for the lemmatizer.
        # someone encountered this problem before.
        # https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
        # There is not a clean way to convert other than to hard code it
        # convert the pos from nltk in a form that works with wordnet.
        words_wordnet_tags = []
        for w in words:
            if w[1] == 'ADJ':
                new_tagged = (w[0], 'a')
            elif w[1] == 'ADV':
                new_tagged = (w[0], 'r')
            elif w[1] == 'NOUN':
                new_tagged = (w[0], 'n')
            elif w[1] == 'VERB':
                new_tagged = (w[0], 'v')
            else:
                new_tagged = (w[0], None)  # if you cannot map it into wordnet then just don't use a POS tag.
                # This means less accurate lemmatizing than if you could label with POS
            words_wordnet_tags.append(new_tagged)

        lemmas = []
        for w in words_wordnet_tags:
            if w[1] != None:
                lemmas.append(self._lemmatizer.lemmatize(w[0].lower(), pos=w[1]))
            else:
                lemmas.append(self._lemmatizer.lemmatize(w[0].lower()))
        return lemmas

    def determine_lemmas(self, text):
        """
        Description:
            returns the lemmas of the words in text .
            Reference this method to have a central location for the vectorization of text.
        Parameters:
            text: a sting you wish to lemmatize
        Returns:
            lemmas: a list of lemmas of the contents of text.
        """
        if len(text) == 0:
            raise ValueError('text is an empty string')
        tagged = self._tag_speech(text)
        after_limiting = self._limit_parts_of_speech(tagged)
        lemmas = self._convert_to_lemma(after_limiting)
        return lemmas
