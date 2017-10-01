# Name : Roshan Zameer Syed
# ID:99999-2920
# description: Compute the Lp distance between all pairs of data points in the data set
# Compute similarity measures between all pairs of data points in the data set

import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.preprocessing import Imputer

data = pd.read_csv('ionosphere.data', header=None)  # Read the data set "Ionosphere"
print('No. of records,attributes: ', data.shape)    # Prints the shape of the Data
print(type(data))                                   # Data type
data_new = data.drop(data.columns[34], axis=1)      # Delete the last column
rows = 351                                          # No of rows 351
total = rows * rows                                 # Total
data_euc = np.arange(total,dtype=np.float).reshape(rows,rows)   #
data_city = np.arange(total,dtype=np.float).reshape(rows,rows)
data_mink = np.arange(total,dtype=np.float).reshape(rows,rows)
data_inf = np.arange(total,dtype=np.float).reshape(rows,rows)
data_cos = np.arange(total,dtype=np.float).reshape(rows,rows)
data_jac = np.arange(total,dtype=np.float).reshape(rows,rows)
data = np.matrix(data_new)                          # Put the data in a matrix
data = data.transpose()                             # Transpose


for i in range(34):                                 # For loop to traverse the data set
    for j in range(34):
        data_euc[i,j] = (distance.euclidean(data[i], data[j]))  # Finding Euclidean distance
        data_city[i,j] = (distance.cityblock(data[i], data[j])) # Finding Manhattan distance
        data_mink[i,j] = (distance.minkowski(data[i], data[j], 3))  # Finding Minkowski distance
        data_inf[i,j] = (distance.chebyshev(data[i], data[j]))      # Finding L = infinity
        data_cos[i,j] = (distance.cosine(data[i], data[j]))         # Finding Cosine similarity
        data_jac[i,j] = (distance.jaccard(data[i], data[j]))        # Finding Jaccard similarity

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)        # Imputer to replace NaN values with mean
imp.fit(data_cos)                                                   # Using fit imputer for Cosine
imp.fit(data_jac)                                                   # using fit for Jaccard
data_clean_cos = imp.transform(data_cos)                            # Transform
data_clean_jac = imp.transform(data_jac)

print('Euclidean distance:', data_euc)                    # Printing the Values
print()
print('Manhattan distance:', data_city)
print()
print('Minkowski distance:', data_mink)
print()
print('Infinity distance:', data_inf)
print()
print('Cosine similarity:', data_clean_cos)
print()
print('Jaggard similarity:', data_clean_jac)