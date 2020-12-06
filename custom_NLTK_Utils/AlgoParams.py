class AlgoParams():
    """
    This is a class of AlgoParams that you are using to see what params create 
    the most accurate classification Algorithms.
    
    """
    def __init__(self, stopWords, NmostFrequent, PosExamples, NegExamples, PartsOfSpeech):
        # the init method will always run in a class
        self._stopWords = stopWords
        self.NmostFrequent = NmostFrequent
        self.PosExamples = PosExamples
        self.NegExamples = NegExamples
        self.PartsOfSpeech = PartsOfSpeech
    
