# author : Roshan Zameer Syed
# id:99999-2920
# description: Principal Component Analysis of the data set "arrhythmia.data"

import pandas as pd
import numpy as np

from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA

data = pd.read_csv('arrhythmia.data', header=None)      # Read data from the file
data.isnull().sum().sum()
data = data.replace('?', np.NaN)                          # Replace missing data with NaN

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)    # Fill missing values with "Mean"
imp.fit(data)
data_clean = imp.transform(data)                                # Transform the data
#print(data_clean)

pca = PCA(n_components=80)
pca.fit(data_clean)
data_red = pca.transform(data_clean)
print("Eigen Values: ", pca.explained_variance_)		# Printing Eigen Values
print("Eigen Vectors: ", pca.components_)			# Printing Eigen Vectors
# print(data_red)
# print (data.shape)
# print(data_clean.shape)
# print(data_red.shape)
print("Variance Ratio: ", pca.explained_variance_ratio_)	# Printing Variance Ratio
print("Sum of the ratio's: ", pca.explained_variance_ratio_.sum()) # Sum of ratio's : 0.996325978866 = 99.6%