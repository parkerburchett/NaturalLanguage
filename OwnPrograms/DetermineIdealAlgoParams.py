"""
Explanation of the goals of this program

When you create each classification algorithm there are several arbritray paramaters 
that you decided on to start. The purpose of this file, and the method that go along with it.
are to decided what version of those params are the most accurate. 


When you look at the data labeling paramaters the params that go into the classifiers

These are the parts

    (
    PositveExamples: A String[] that is read in from the file containing every positive reivew
    NegativeExamples: A String[] read in from the negative examples
    
    
    documents : a list of tuples containging An element of (PositiveExamples, "Pos / Neg")
                             PostiveExamples[i] is a string here.
    
    all_words[] : a frequency distribution of how often each word occurs in the entire sample
        You can do things to limit what types of words are considerd here.
                limit_features() serves this purpose
                
    word_features: the N most common words, where you can change N. In the entire sample
    
    
    feature_sets: A list of Dictionary, String tuples 
    representing the boolean of if a word is in the common features and if the review is postiive or negative 
     
    N mostCommon Words
    
    
    Regex of what Parts of speech to consider.
    
    Boolean to consider stopwords.
     )


should create a data object with these parts:


from  custom_NLTK_Utils.README.txt

For Each , With_stopWords and WithoutStopWOrds: (2)
    For All combinations of (Adjectives, Adverbs, Verbs): (6)
        For each size of word_features 2000, 3000, 4000 (3):
            Generate a new FeatureSet based on the above Params
            Train the 5 Algos
            Train the VoteClassifer
            Write to a File the accuracy of the 5 methods 
            Write to a File the accuracy of the VoteClassifer
            Write to a file lookAtAccuracy() for the Vote CLassifier
            Save as a tuple (RegexOFPosToConsider, SizeOf WordFeatures, 
            Accuracy of VoteClassifer, ALL,
            Vote Clasifer Accuracy at 60% confident, Accuracy at 80%, Accuracy at 100% )
            Num testing set at 60% confident, num at 80% confident and num at 100% confident
                Write that tuple to a different

"""