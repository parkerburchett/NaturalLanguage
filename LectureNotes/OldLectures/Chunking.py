#https://www.youtube.com/watch?v=imPpT2Qo2sk&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL&index=5

"""
Named entity is the thing (object) or person that a sentence is talking about
This is a subject. They are generally a noun.

Words that Modify a particular noun.

Chunking is the way of breaking apart sentences into meaningful chunks.

The genral way to go about doing this is first to assign parts of speech to 
every word in the text. using regualar expressions. and the labels of parts of 
speech to convert them into meaningful chunks.

It follows this format:
CC coordinating conjunction
CD cardinal digit
DT determiner
EX existential there (like: “there is” … think of it like “there exists”)
FW foreign word
IN preposition/subordinating conjunction
JJ adjective ‘big’
JJR adjective, comparative ‘bigger’
JJS adjective, superlative ‘biggest’
LS list marker 1)
MD modal could, will
NN noun, singular ‘desk’
NNS noun plural ‘desks’
NNP proper noun, singular ‘Harrison’
NNPS proper noun, plural ‘Americans’
PDT predeterminer ‘all the kids’
POS possessive ending parent’s
PRP personal pronoun I, he, she
PRP$ possessive pronoun my, his, hers
RB adverb very, silently,
RBR adverb, comparative better
RBS adverb, superlative best
RP particle give up
TO, to go ‘to’ the store.
UH interjection, errrrrrrrm
VB verb, base form take
VBD verb, past tense, took
VBG verb, gerund/present participle taking
VBN verb, past participle is taken
VBP verb, sing. present, known-3d take
VBZ verb, 3rd person sing. present takes
WDT wh-determiner which
WP wh-pronoun who, what
WP$ possessive wh-pronoun whose
WRB wh-adverb where, when

https://pythonprogramming.net/regular-expressions-regex-tutorial-python-3/
Link to the regex stuff
"""

"""
https://www.youtube.com/watch?v=EymPQgCtcAE&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL&index=6
Chinking is the next step of chunking.

You Chink a Chunk

Chinking is the removal of something
It says X is a chunk unless if it has X.

Make X into a chunk that fits this pattern as long as there is not Y inside of 




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
        for i in tokenized[50:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r"""CHUNK: {<.*>+}
                                   }<VB.?|IN|DT|TO>+{"""
            # the syntax for a chink is }{ outward facing brackets. This says 
            # dont' include this as a chunk
            chunkParser =  nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            chunked.draw()
            # <> means part of speech tag the other symbols are regex
            # 
    except Exception as e:
        print(str(e))


process()
