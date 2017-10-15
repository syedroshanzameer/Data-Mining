"""Name : Roshan Zameer Syed
ID : 99999-2920
Project : Kmeans clustering n=3 on Iris data set """

# packages
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.patches as mpatches

# Loading the dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
print('Target values: ', y)

kmeans = KMeans(n_clusters=3).fit(X)
labels = kmeans.labels_
labels = labels.transpose()
print('Labels:', labels)
centroids = kmeans.cluster_centers_
print('Centers:', centroids)

# Confusion Matrix
confusion = confusion_matrix(y, labels)
print('Confusion Matrix: \n', confusion)

# Cross Table
table = pd.crosstab(y, labels)
print('Cross table: \n', table)

print(classification_report(y, labels))

petalWidth = X[:, 3]
sepalLength = X[:, 0]
petalLength = X[:, 2]

# Plotting 2d Set of 3 clusters
colormap = np.array(['Red', 'Blue', 'Green'])
plt.figure(figsize=(16,5))
plt.title('kmeans = 3 Clusters')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.colors()
plt.subplot(1,2,1)
plt.scatter(petalWidth,sepalLength, c=colormap[labels], s=30)
plt.title('Sepal Length vs Sepal Width')

plt.subplot(1,2,2)
plt.scatter(petalLength, petalWidth, c=colormap[labels], s=30)
plt.title('Petal Length vs Petal Width')

red = mpatches.Patch(color='red', label='The Petal Width')
blue = mpatches.Patch(color='blue', label='The Sepal Length')
green = mpatches.Patch(color='green', label='True Values')
plt.legend(handles=[red,blue,green])
plt.show()

