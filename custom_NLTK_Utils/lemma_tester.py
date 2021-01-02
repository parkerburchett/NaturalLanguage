# this is where I am debugging CustomLemmatizer.py
import CustomLemmatizer
import datetime


my_lemmatizer = CustomLemmatizer.CustomLemmatizer()

sent = 'Within this article we’ll explore the ImportError and ModuleNotFoundError in a bit more detail, beginning with where they sit in the overall Python Exception Class Hierarchy. '

start = datetime.datetime.now()

for i in range(10000):
    a = my_lemmatizer.determine_lemmas(sent)

end = datetime.datetime.now()

print(end-start)