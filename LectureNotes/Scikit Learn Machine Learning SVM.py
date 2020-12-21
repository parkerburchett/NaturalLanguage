"""
Lecture on support vector machines

source: https://www.youtube.com/watch?v=KTeVOb8gaD4


"""

from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt

digits = datasets.load_digits()

# it is best to have all of the data be boolean or in range -1 to 1 as a float

# print(digits.data)
# print(digits.target)
# print(len(digits.data))
# print(digits.images[0])
#  gamma is the learning rate.
# You can think of it as step size of gradient descent
clf = svm.SVC(gamma=.00001,C=100,probability=True)

x,y = digits.data[:-10], digits.target[:-10]

clf.fit(x,y)

# plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation="nearest")
# plt.show()

# works
print('Prediction: {} \nCorrect:    {}'.format(clf.predict(digits.data[-10:]), digits.target[-10:]))