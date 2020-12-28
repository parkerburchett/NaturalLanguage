"""
This is where I am testing methods to explain certain why a prediction was reached.

# you should look at # SGDClassifier.coef. I am unsure how to get that. right now you can look at in the debugger

# you might want to use decision_function()

"""

from NaturalLanguage.custom_NLTK_Utils import Pickle_Utils


def create_feature_coef_list(word_features, coef):
    """
    This method returns the word_feature, coef pair

    It shows the coeffeicnt assigned to each word_feature



    """
    pairs = []
    for i in range(len(word_features)):
        a_pair = (word_features[i],coef[0][i])
        pairs.append(a_pair)  # might be better as a dictionary. It will be better as a dictionary. Change later
    return pairs

my_voteClassifier = Pickle_Utils.unpickle_this(r'TrainedVoteClassifier_N6000.pickle')


a_classifier = my_voteClassifier.get_classifier_list()[0]

coef = a_classifier.coef_
word_features = my_voteClassifier.get_word_features()

pairs = create_feature_coef_list(word_features,coef)
# I am wrong here. Weights do not neccessiraly corrospond to importance. 
# read this https://machinelearningmastery.com/calculate-feature-importance-with-python/


for i in range(10000):
    if(pairs[i][1]<-5):
        print(pairs[i])


print('fin')

