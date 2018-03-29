"""Name : Roshan Zameer Syed
ID: 99999-2920
Project: Data Mining Project4


# Example data has been downloaded from the open access Human Gene Expression Atlas and represents typical data
#  bio informaticians work with.

# It is "Transcription profiling by array of brain in humans, chimpanzees and macaques, and brain, heart,
#  kidney and liver in orangutans" experiment in a tab-separated format.

# Importing packages
import numpy as np
import matplotlib.pyplot as plt
#from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram
from fastcluster import *

# Loading data from a text file ,names - field names are read from the first valid line,data type as float and with delimiter '\t'
data = np.genfromtxt("ExpRawData-E-TABM-84-A-AFFY-44.tab", names=True,usecols=tuple(range(1,32)),dtype=float, delimiter="\t")

# printing the number of elements in the list
print(len(data))

# printing the number of elements in name
print(len(data.dtype.names))

# Viewing array in a different type and data type
data_array = data.view((np.float, len(data.dtype.names)))

# Transpose the data_array
data_array = data_array.transpose()

# Printing data_array
print(data_array)

#data_dist = pdist(data_array) # computing the distance

# perform hierarchical clustering on the matrix data_array with metric "Euclidean"
data_link = linkage(data_array,method="single",metric="euclidean") # computing the linkage


# Printing the linkage steps
print('Linkage Steps: \n', data_link)

# creating a dendrogram with labels
dendrogram(data_link,labels=data.dtype.names)

# Plotting with lables on x-axis "Samples" and y-axis "Distance"
plt.xlabel('Samples')
plt.ylabel('Distance')

# Title of the plot as "Bottom-up clustering"
plt.title('Bottom-up clustering', fontweight='bold', fontsize=14);

# printing the plot diagram
plt.show()