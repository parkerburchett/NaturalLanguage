"""
https://www.youtube.com/watch?v=nla4C-VYNEU&t=2s
Created on Sat Nov 22 17:15:22 2020

@author: parke


SciKit-Lean incorporation

SciKit learn is a machine learning toolkit.

This is a lecture on how to integrate SK Learn into NLTK toolkit. 

First I had to learn the internals of the different Naive Bayes Algorithms. 
Those notes are below.

There also using Support Vector Machines to classify it as well. 

I dont know about the internals of those but I will learn about them in the future




_______________________________Overview______________________________________






_____________What is the Multinomial Naive Bayes?____________________

SOURCE:  https://www.youtube.com/watch?v=O2L2Uv9pdDA

    This is my notes from the youtube lecture above. It walks through the terms
    and the math for what the multinomial Naive Bayes does. 
    In short the multinomial Naive Bayes looks at the frequency distribution of Positive Reviews
    and negative Reviews then it says, based on the bag of words in the Unknown Reviews. 
    Then it looks at each word add does Bayes Theorem on it to decide if the word is more likly in Positive or Negative

Start with getting a frequency distribution of all the words in the Positive Reviews.

From that you can get the probability of the word "Great" occurring given
that it is a Positive Review. (for the example assume P("greatOccuring"| Positive Review)) 

You would write that like P(The word "GREAT" occurs in a review | it is a positive review)
Do this for all of the words in the positive reviews. Or the most frequent 3000 words.

Bayes's Therum lets you swap P(B|A) into P(A|B)

This is valuable since you know that in Positive Review the word "GREAT" occurs often
So when you encounter "Great" in a Unknown Review, the fact that it often is in Positive reviews
Inclines you to think that the message is positive rather than negative. 

Do the same frequency distribution for all the negative Reviews and their words

Here is the process for the

Step 1) start with the odds that a Review is Positive. You can get this from the training data
P(PositiveReview) = .5 for my example it is half the training data
The initial guess, knowing no other information, that a message is positive is called a Prior probability

This where the term Priors comes from
You can remember this because it says 
    "A priori I expect that the message is positive this % of the time."

For every word in the reivew 

P(Positive Review)* P(word_1|PositiveReview) *P(word_2|PositiveReview) = ScoreIfPositive

Do the same thing but with the Frequency Distribution of words in the negative

P(NegativeReview)* P(word_1|NegativeReview) *P(word_2|NegativeReview) = ScoreIfNegative

Do this for every word. If a word occurs more than once you multiply it into the product more than once

If ScoreIfPositive > ScoreIfNegative we guess that the Review is Positive 

else if ScoreIfPositive < ScoreIfNegative we guess that the Review is Negative

Note:
    There is a problem when a word occurs 0 times in the training data. 
    In this case the P(Word that occurs 0 times| Anything) =0
    This causes you to misclassify things. 
    
    To solve this problem simply add a constant to the count of each word frequency.
    This is called "alpha"

The reason why this is Naive is that it ignores word orders

    eg getScore("Great Movie") == getScore("Movie Great").
    
    It treats a Language like a bag filled with words and a message
    like a collection of unordered words.
    
    Since it ignores the relationship between words. The Naive Bayes is 'HIGH BIAS'

_______________________End of Multinomial Naive Bayes_________________________





_______________________What is Gaussian Naive Bayes?__________________________

Gaussian - This word means that there is a normal, bell curve, distribution

Source: https://www.youtube.com/watch?v=H3EjCKtlVog

This is the notes from the lecture above

You are trying to infer, from a set of data, if a person likes Taylor Swift.

You have two groups LikesTaylor and NotLikesTaylor.

You also know the mean, and standard deviation, of a view data about each group.
These are all continuous variables 

You have oz of coffee drank per day

How long have they had netflix days

How tall are they, inches

For each of these variables you get a Mean and Std Dev for each group and each variable.

Now You have a Particular person, with values for each of those variables and attempt to 
determine if they like Taylor Swift or not. 

Step 1. Get the A priori odds that a person likes Taylor Swift. Estimate this from the training data
    These are called prior probabilities
    
    P(LovesTaylor) 
    *Likelihood(Height = 6 feet | LovesTaylor)
    *Likelihood(Coffee = 12oz| LovesTaylor)
    *Likelihood(netflix = 500Days | LovesTaylor)
    
Sometimes the likelihoods are very small. Too small for a computer to accurately keep track of. 
to solve this take the log of every value. Usually you take the natural log of each value. 

Think about this formula as getting the score of Loves Talyor

getScore LovesTaylor(PersonA).

You do this also for if they do not love Taylor swift. 

getScore Not Love Taylor(PersonA)

if  getScoreNotLovesTaylor(PersonA) >getScoreLovesTaylor(PersonA)
    We will classify them as Not Loving Taylor Swift
    
else if getScoreNotLovesTaylor(PersonA) < getScoreLovesTaylor(PersonA)
    we will classify them as Loving Taylor Swift.
    
    Since ln(anything less than 1) is always a negative number. 
    We classify based on what is closer to zero. 
    Whatever score is closer to zero we think is the proper classification


This will later down the line let you see what variables let you classify the most accurately,

Eg knowing less variables how will that let you distingiush people better. 

This is useful since it reduces the amount of computational work needed and 


_______________________End of Gaussian Naive Bayes__________________________




________________________What is the Bernoulli Naive Bayes______________________
Source: 
https://iq.opengenus.org/bernoulli-naive-bayes/#:~:text=This%20is%20used%20for%20discrete,or%201%20and%20so%20on.
Bernoulli -This word means there is a discrete distribution of True or False.
This only works with binary data. 

Eg you have 3 boolean variables and you are trying to predict another boolean outcome

Step1. get a priori values for the outcome. 

get prob of Pass | studied = TRUE  

Do this for all the values.

The compare the prob of Pass given the variables, and the prob of not pass given the variables. 

The prob that is larger is the guess. 

_______________________end Bernoulli Naive Bayes_______________________________



____________________General Notes____________________________________________

Bayes theorem: P(A|B) = (P(B|A)P(A)) / P(B)

What this means in plain english is

P(A|B) This is a statement of conditional probability that goes like that
P(A|B) = The probability of A given B has already occured. 

(P(B|A)P(A)) / P(B)

This statement means
(The probability of B, given that A has already occured ) * the probability of A
 
All devided by the probability of B is equal to the odds 
    of A occuring given B has already occured. 

    From the sklearn.naive_Bayes.MultinomialNB api it does this 
    "alphafloat, default=1.0
    Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing)."
    
_______________________________end General Notes______________________________

"""





import random, pickle, nltk

from nltk.corpus import movie_reviews
# you need to wrap a SKlearn Classifer in a NLTK classifer wrapper so that you 
# can treat them all as the same type to get better at predicting them
from nltk.classify.scikitlearn import SklearnClassifier
# these are different classifing algorithms
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier

#this is support vector machines.
# I have no Idea what this is
from sklearn.svm import SVC, LinearSVC, NuSVC


def find_Features(document):
    words = set(document) 
    features = {}
    for w in word_features:
        features[w] = (w in words) # this a boolean
    return features  


documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)
all_words =[]
for w in movie_reviews.words():
    all_words.append(w.lower())   
all_words = nltk.FreqDist(all_words) 
word_features = list(all_words.keys())[:3000]


   
    
# I need a clearer understanding of creating sets like this
feature_sets = [(find_Features(rev), category) for (rev, category) in documents]

TrainingSet = feature_sets[:1900]
TestingSet = feature_sets[1900:]

classifier_NaiveBayes = nltk.NaiveBayesClassifier.train(TrainingSet)
print("Original Accuracy of Naive Bays in Percentage:", 
      (nltk.classify.accuracy(classifier_NaiveBayes, TestingSet)*100))



MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(TrainingSet)
print("Accuracy of MNB_classifier in Percentage:", 
      (nltk.classify.accuracy(MNB_classifier, TestingSet)*100))



BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(TrainingSet)
print("Accuracy of BernoulliNB_classifier in Percentage:", 
      (nltk.classify.accuracy(BernoulliNB_classifier, TestingSet)*100))

#___________________________________________________________________
# LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
# LogisticRegression_classifier.train(TrainingSet)
# print("Accuracy of LogisticRegression_classifier in Percentage:", 
#       (nltk.classify.accuracy(LogisticRegression_classifier, TestingSet)*100))


SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(TrainingSet)
print("Accuracy of SGDClassifier_classifier in Percentage:", 
      (nltk.classify.accuracy(SGDClassifier_classifier, TestingSet)*100))


SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(TrainingSet)
print("Accuracy of SVC_classifier in Percentage:", 
      (nltk.classify.accuracy(SVC_classifier, TestingSet)*100))


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(TrainingSet)
print("Accuracy of LinearSVC_classifier in Percentage:", 
      (nltk.classify.accuracy(LinearSVC_classifier, TestingSet)*100))

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(TrainingSet)
print("Accuracy of NuSVC_classifier in Percentage:", 
      (nltk.classify.accuracy(NuSVC_classifier, TestingSet)*100))












