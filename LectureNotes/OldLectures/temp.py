# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 21:55:43 2020

@author: parke
"""

f = open("demofile2.txt", "w")


def foo(counter, f):
    
    f.write(str(counter)+ "\n")
    
    foo(counter+1, f)
    
foo(0,f)
