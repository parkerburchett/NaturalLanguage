import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer as ps


traintext = state_union.raw("2005-GWBush.txt")
mytext = state_union.raw("2006-GWBush.txt")

# this trains the sentence tokenizer on the 2005 speech
custom_Tokenizer = ps(traintext)

tokenized = custom_Tokenizer.tokenize(mytext)


def process():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)
    except Exception as e:
        print(str(e))


process()
# this makes a serios of tuples that looks like:
# for every word
#'confident', 'NN'), ('of', 'IN'), 
#('the', 'DT'), ('victories', 'NNS'), 
#('to', 'TO'), ('come', 'VB')
#