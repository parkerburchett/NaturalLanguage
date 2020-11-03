#https://www.youtube.com/watch?v=imPpT2Qo2sk&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL&index=5

"""

https://www.youtube.com/watch?v=LFXsG7fueyk&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL&index=7


A named entitity you can think of a particular thing.

NOTE: the named entity tracker is very bad for NLTK


You might get better outcomes if you just look for nouns
"""





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
        for i in tokenized[5:10]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged)
            namedEnt.draw()

    except Exception as e:
        print(str(e))


process()
# -*- coding: utf-8 -*-

