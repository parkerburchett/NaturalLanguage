"""
NLTK has a whole library of sample texts that you can do stuff with

It is easier to navigate NLTK on your own computer. 





"""
from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

sample = gutenberg.raw("bible-kjv.txt")

tok = sent_tokenize(sample)

for i in tok[5000:5100]:
    print(i)