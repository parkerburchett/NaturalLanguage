"""
I was trying to use a LinearSVR() classifier and found that I did not understand the numpy code.

This is a lecture where I am learning the basics of numpy.
Souce: https://www.youtube.com/watch?v=QUT1VHiLmmI

Multi Dimensional Array Library

Much faster than list manipulation.
You can cast ints as using less bytes in memory by saying things like int16 or int 32.

An integer in a list has these parts

size 4 bytes
reference count 8 bytes
object type 8 bytes
object value 8 bytes

You have to load less things into memory.

You don't have to do type checking.

Numpy uses continuous memory.

This uses the cache more efficiently

This is a matlab replacement.

    This library is the backbone of a whole bunch of different libraries like Pandas and much of machine learning


If you pass in a int value larger than what the memory can hold, it wraps around and is treated as mod( intValue)

Get specific cell A[rowIndex, colIndex]

get an entire row A[0,:]
    This will get the entire row

[startIndex:endIndex:stepSize]

"""
import numpy as np

prob =257
aList = [3,prob,2,0]
a = np.array(aList)
b = np.array([2,3,2,1])

c =[('bob',int(12)),('steve','13')]


cA = np.array(c)
b = np.array([2,3,2,1])
zeros = np.zeros((5,5,5,5))

fullTest =np.full((5,5), 8)

full_likeTest = np.full_like(b,9)

l = fullTest[0,:]

r = np.random.rand(4,2)
r2 = np.random.random_sample(a.shape)
r3 = np.repeat(r2,1, axis=0)
##########

c =np.dot(a,b)


a = np.full((9,2), 4)
b = np.full((2,4), 2)
c = np.matmul(a,b)

##########

a = np.random.randint(0,10,(3,3))
b =np.median(a)
c = np.sum(a)



##############

b = a.reshape(1,9)



############# Vertically Stacking Matrixes


c = np.vstack([b,b,b])


############ Load data from a file


# use np.genfromtxt('filename.txt', delimiter=',')

############# fancy indexing

# only get the values greater than 6 inside of A
c = a[a>6]
# get all the values greater than 3 in a

print(a)
print(b)
print(c)
