# -*- coding: utf-8 -*-
# ID: 99999-2920
# Name: Syed Roshan Zameer
# Assignment 2: Standardization of the Data

import numpy

X1 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
X2 = [300, 500, 1000, 2000, 3500, 4000, 4300, 6000, 2500, 2700]

# Calculating �New Age"
new_Age =(X1-numpy.mean(X1))/numpy.std(X1)

# Calculating �New Income"
new_Income = (X2-numpy.mean(X2))/numpy.std(X2)

# Print the new Data set
print ("The new Age is : ", new_Age.round(3,None))
print ("The new Income is :", new_Income.round(3,None))





