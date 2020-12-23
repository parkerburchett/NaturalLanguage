from NaturalLanguage.custom_NLTK_Utils import dataLabeling as dl
from NaturalLanguage.custom_NLTK_Utils import AlgoEvalutationUtils as AE
from NaturalLanguage.custom_NLTK_Utils import AlgoParams
from NaturalLanguage.custom_NLTK_Utils import Pickle_Utils as ip

def main():
    print('You have started')
    shortPos = open(
        "C:/Users/parke/Documents/GitHub/NaturalLanguage/NaturalLanguage/OwnPrograms/short_reviews/shortPositive.txt",
        "r").read()
    shortNeg = open(
        "C:/Users/parke/Documents/GitHub/NaturalLanguage/NaturalLanguage/OwnPrograms/short_reviews/shortNegative.txt",
        "r").read()

    my_param = AlgoParams.AlgoParams(True, 2000, shortPos, shortNeg, "ADJ")
    all_words = dl.assemble_all_words(my_param)
    word_features = dl.assemble_word_features(all_words, my_param)
    feature_sets = dl.create_feature_sets(my_param)
    classifiers = AE.createAndTrain_Classifiers(feature_sets, split_train=False)
    my_vote_classifier = classifiers[5]
    # AE.writeAlgoEvaluation(my_param, classifiers, feature_sets, fileName="ModelResults_ToUseOnTwitter_ADJ")
    ip.pickle_this(word_features, "TwitterModel_word_features_N2000_ADJ")
    ip.pickle_this(my_vote_classifier, "VoteClassifier_N2000_ADJ")
    print('finished')

main()