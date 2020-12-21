#https://www.youtube.com/watch?v=T68P5-8tM-Y

"""
Wordnet

Look up synonomyes, def aand antypnomes of words.


synset 
Synset('hardened.s.05')

first entry is a synonmyn second entry is the part of speech
 third entry 
is the number in the set. 

you also can look at sematic similarity 


"""

from nltk.corpus import wordnet

sysn = wordnet.synsets("set")

# this is how you see all the words that are there
#for in sysn[10:20]:
   #print(i.lemmas()[0].name())
  # print(i)
  # print(i.definition())
  # print(i.examples())
  
antonyms =[]
synonyms = []

for syn in wordnet.synsets("fair"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            #boolean has antnomys
            antonyms.append(l.antonyms()[0].name())

#print(set(antonyms))
#print("")
#print(set(synonyms))

w1 = wordnet.synset("ship.n.01") # get the first entry as a noun of ship
w2 = wordnet.synset("dog.n.01")
how_similar = w1.wup_similarity(w2)
print(w1.name(), " and ",w2.name()," are " ,how_similar," percent similar")
