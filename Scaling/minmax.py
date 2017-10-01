# -*- coding: utf-8 -*-
# ID : 99999-2920
# Name : Roshan Zameer Syed
# Assignment 2: Min Max Scaling 

import numpy

X1 = [12,14,18,23,27,28,34,37,39,40]
X2 = [300,500,1000,2000,2500,2700,3500,4000,4300,6000 ]

age_min = numpy.min(X1)
age_max = numpy.max(X1)
new_min = 0
new_max = 1
income_min = numpy.min(X2)
income_max = numpy.max(X2)

# Calculating �New Age" with the range of [0,1]
new_age = (((X1-age_min)*(new_max-new_min))+new_min)/(age_max-age_min)

# Calculating �New Income" with the range of [0,1]
new_income = (((X2-income_min)*(new_max-new_min))+new_min)/(income_max-income_min)

# Printing the new Data Set
print ("The new Age is : ",new_age.round(3,None))
print ("The new Income is :",new_income.round(3,None))