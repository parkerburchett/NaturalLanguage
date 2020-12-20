import pickle
import NaturalLanguage.custom_NLTK_Utils.dataLabeling as dl

# now using a subset of the labeled twitter data

VoteClassifierIN = open(r"C:\Users\parke\Documents\GitHub\NaturalLanguage\NaturalLanguage\OwnPrograms\Kaggle_LinearSupportVectorRegression.pickle", "rb")
myClassifier = pickle.load(VoteClassifierIN)
VoteClassifierIN.close()

word_featuresIN = open(r"C:\Users\parke\Documents\GitHub\NaturalLanguage\NaturalLanguage\OwnPrograms\KaggleN5000_word_features.pickle", "rb")
default_word_features = pickle.load(word_featuresIN)
word_featuresIN.close()


def determine_sentiment(text, long=False):
    features = dl.find_Features(text, default_word_features)
    ans = myClassifier.classify(features)

    if long:
        long_ans = "The model considered: \n{}\nBased on these words:\n" \
                   "'{}'\nIt thinks the sentiment is {}".format(text, whatWordsMattered(text), ans)
        return long_ans
    else:
        return ans

def whatWordsMattered(text, long=False):
    features = dl.find_Features(text, default_word_features)
    items = features.items()

    words =[]
    for w in items:
        if w[1] == True:
            words.append(w[0])
    if long == False:
        return words
    else:
        return "These were the words treated as features {}".format(str(words))