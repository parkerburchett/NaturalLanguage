from NaturalLanguage.custom_NLTK_Utils import dataLabeling as dl
import numpy as np

def Main():
    test_string = "I was walking down the street yesterday and I saw a man on the ground"
    word_features = np.array(['was', 'down', 'I', 'cow'])

    vector = dl.text_to_vector(test_string,word_features)
    print(vector)

    bag_of_words = dl.vector_to_words(vector,word_features)
    print(bag_of_words)


Main()