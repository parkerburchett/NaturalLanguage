from NaturalLanguage.custom_NLTK_Utils import dataLabeling as dl
import NaturalLanguage.OwnPrograms.Kaggle_Program.Vectorize_Kaggle_Data as VKD
import numpy as np

def Main():

    # you need to step through every part of the method here.

    fileLocation = open(r"C:\Users\parke\Documents\GitHub\NaturalLanguage\NaturalLanguage\Datasets\tinydocs.txt","r")
    # Test the vectorization of targets and sentences.
    documents = dl.assemble_kaggle_documents(fileLocation)
    # you need to remove puncuation since this over counts single and double quotes for whatever reason
    word_features = VKD.get_word_features(5, default=False)

    # create vectors and targets
    vectors, targets = VKD.convert_docs_to_vectors(documents,word_features,5)

    print(word_features)



    test_string = "I saw a silly man cry the way"
    test_vector = dl.text_to_vector(test_string,word_features)
    print(test_vector)
    bag_of_words = dl.vector_to_words(test_vector,word_features)
    print(bag_of_words)

    # at this point you are certain that you can get the most frequent words (ignore the punctuation problems)

    # you can convert any string into a vector and back into a bag of words

    # you can correctly create word_features, vectors and targets.

    


Main()