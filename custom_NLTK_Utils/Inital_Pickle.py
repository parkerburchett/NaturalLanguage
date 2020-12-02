import pickle
import dataLabeling


def customPickle(thingToPickle, objectName): 
    """
        Pass this an Object and it will create a pickled instance of it
        it will be save in this format "pickled_"+ name +".pickle"
        
        eg You want to pickle a list called myList
        
        this will create a file called 
        
        myList.pickle in your current dirictory
    """

    objectName = objectName +".pickle"
    outLocation = open(objectName, "wb")
    pickle.dump(thingToPickle, outLocation)
    outLocation.close()

def pickle_Training_Params(PositiveExamples, NegativeExamples):
    """
       This calls the proccesses in dataLabeling.py 
       but saves the pickled resutls instead
       You also need to save somehow what the limit_features Method you are using. 
       // Maybe you need
    """
    documents = dataLabeling.assemble_Documents(PositiveExamples, NegativeExamples)
    all_words = dataLabeling.assemble_all_wordsFRQDIST(PositiveExamples,NegativeExamples)
    word_features = dataLabeling.assemble_word_features(all_words, 3000)
    feature_sets = dataLabeling.create_feature_sets(PositiveExamples,PositiveExamples)
    
    customPickle(documents,"documents")
    customPickle(all_words,"all_words")
    customPickle(word_features,"word_features")
    customPickle(feature_sets,"feature_sets")
    

def pickle_Algos(ListOfTrainedAlgos):
    
    