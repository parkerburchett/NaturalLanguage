import pickle
import NaturalLanguage.custom_NLTK_Utils.dataLabeling as DL

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
