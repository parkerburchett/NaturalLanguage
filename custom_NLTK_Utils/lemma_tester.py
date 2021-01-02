# this is where I am debugging CustomLemmatizer.py
import CustomLemmatizer
import datetime

start = datetime.datetime.now()
my_lemmatizer = CustomLemmatizer.CustomLemmatizer()

sent = 'Here in a bit more detail, beginning with where they sit in the overall Python Exception Class Hierarchy. '


for i in range(100000):
    a = my_lemmatizer.determine_lemmas(sent)

end = datetime.datetime.now()

print(end-start)