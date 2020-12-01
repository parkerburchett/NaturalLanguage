This is where I am keeping the methods I expect to use for NLTK stuff in the future



Right now my sentiment tester is very bad in the wild. 

I need to improve it in several ways.


I need to only train it on some parts of speech.

    Try only Adjectives, Adverbs and Verbs.
    
    Try using a differnet size of wordFeatures. say 2000, 3000, 4000
    
    Need to try only 10% of the data as a testingSet.
    
    I to remove redundent words, or POS when I pass it text 
    sentiment(text)
    
    I need to compare these results in a method to see how and where thery are different
    


Here is what the psudoCode could look like:

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
                
                
                
This is only 36 calls, Each call takes approx 3 Minutes.

Once you do this to can have some idea about what parameters are the best for the testing sets
