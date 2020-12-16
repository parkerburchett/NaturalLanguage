import pickle
import os

def customPickle(thingToPickle, object_name):
    """
        Pass this an Object and it will create a pickled instance of it
        it will be save in this format "pickled_"+ name +".pickle"
        
        eg You want to pickle a list called myList
        
        this will create a file called 
        
        myList.pickle in your current directory
    """
    object_name = "./" + object_name + ".pickle"
    out_location = open(object_name, "wb")
    pickle.dump(thingToPickle, out_location)
    out_location.close()
