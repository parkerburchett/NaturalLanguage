
"""

https://www.youtube.com/watch?v=uoHVztKY6S4&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL&index=8


lemmatizing.
This is similar to stemming, but the output is a real word.

This gives you a similar word with the same meaning.

In general, lemmatizing is better (according to the lecture guy) than stemming


the default is that the word you are lemmatizing is a noun
if you don't want it so be a noun you need to tell it that
ege
print(lem.lemmatize("run","v")) run as verb
print(lem.lemmatize("run","n")) run as noun
print(lem.lemmatize("run")) run as noun

"""


from nltk.stem import WordNetLemmatizer

lem = WordNetLemmatizer()

print(lem.lemmatize("better", pos="a"))
print(lem.lemmatize("best", pos="a"))
print(lem.lemmatize("run","v"))
print(lem.lemmatize("run","n"))



def process():
    try:
        print("sl")

    except Exception as e:
        print(str(e))



# -*- coding: utf-8 -*-

