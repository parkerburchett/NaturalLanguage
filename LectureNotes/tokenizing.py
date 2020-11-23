# https://www.youtube.com/watch?v=FLZvOKSCkxY
# this is series of videos about how to use national
# porter stemmer is a old english find the stem of a word algorithm
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#corpora is a body of text
# coudl be meidicla journals
# presidential speeches
# a corpra is a set of (written works that are similar in some way)

# lexicon
# The words and their meanings
# investor speak vs regualr english
# lexicon is the words and their meanings within a corpra

"""
stops = set(stopwords.words("english"))
# stop words are words that don't mean anything
words = word_tokenize(a)

filtered = []

for w in words:
    if(w not in stops):
        filtered.append(w)
print(filtered)

"""
# steming
# data preprocessing
# look at the root stem of a word. 
# this reduces down to the object itself.
# since you can refer to the same object in half a dozen differnet spellings
# this makes it so that they all point to the same thing


a = "He was seen as a champion of individualism and a prescient critic of the countervailing pressures of society. He disseminated his thoughts through dozens of published essays and more than 1,500 public lectures across the United States."

ps = PorterStemmer()

w = {"reading", "reader", "read", "readly"}

for i in w:
    print(ps.stem(i))
