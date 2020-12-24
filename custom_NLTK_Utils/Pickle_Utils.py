import pickle


def pickle_this(thing_to_pickle, object_name):
    """
        Parameters:
            thing_to_pickle: A python object you with to pickle
            object_name: the name you with to pickle this under
                will look like "./object_name.pickle"
    """
    object_name = "./" + object_name + ".pickle"
    out_location = open(object_name, "wb")
    pickle.dump(thing_to_pickle, out_location)
    out_location.close()

def unpickle_this(file_location):
    """
        Parameters: file_location a raw string the absolute path of the pickled python object you want to load into memory

        Returns:
            my_object: the now unpickled python object. Note this can be any type of object.
    """
    file_in = open(file_location,"rb")
    my_object = pickle.load(file_in)
    file_in.close()
    return my_object
