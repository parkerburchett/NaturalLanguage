import pickle
import NaturalLanguage.custom_NLTK_Utils.dataLabeling as DL

def customPickle(thingToPickle, object_name):
    """
        Pass this an Object and it will create a pickled instance of it
        it will be save in this format "pickled_"+ name +".pickle"
        
        eg You want to pickle a list called myList
        
        this will create a file called 
        
        myList.pickle in your current dirictory
    """
    object_name = "./"+ object_name +".pickle"
    file_name = open(object_name)
    file_name.truncate(0)
    file_name.close()

    out_location = open(object_name, "wb")
    pickle.dump(thingToPickle, out_location)
    out_location.close()
