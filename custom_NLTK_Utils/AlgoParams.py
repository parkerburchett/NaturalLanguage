class AlgoParams():
    """
    This is a class of AlgoParams that you are using to see what params create 
    the most accurate classification Algorithms.
    
    """
    def __init__(self, the_stop, NmostFrequent, PosExamples, NegExamples, PartsOfSpeech):
        # the init method will always run in a class
        self.the_stop = the_stop
        self.NmostFrequent = NmostFrequent
        self.PosExamples = PosExamples
        self.NegExamples = NegExamples
        self.PartsOfSpeech = PartsOfSpeech
    
