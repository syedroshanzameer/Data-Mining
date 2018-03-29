"""
Name : Roshan Zameer Syed
ID: 99999-2920
Project 7 : Support vector machine for the face classification problem
"""
from sklearn.datasets import fetch_lfw_people
import pandas as pd
import numpy as np

faces = fetch_lfw_people(min_faces_per_person=60)   #Importing the data set with min faces = 60
n_samples, h, w = faces.images.shape
print('Target names: ', faces.target_names)                           #Printing the target names
print('Shape of the data: ', faces.images.shape)                           #Shape of the data
X = faces.data
#print(X)
print(faces.data.shape)
n_features = faces.data.shape[1]                    # features is the dimension
print(n_features)

y = faces.target
print(y)
target_names = faces.target_names
n_classes = target_names.shape[0]
print(n_classes)

print("n_samples: %d" % n_samples)                  # Print number of samples
print("n_features: %d" % n_features)                #Print number of features
print("n_classes: %d" % n_classes)                  #Print number of classes

# Splitting the data set to training and testing data set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)        #Split the dataset to training and Test

print('Before Dimension Reduction: ', X_train.shape)            # Dataset before dimension reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=200)                                     # Using PCA for dimension reduction
pca.fit(X_train)                                                # Fit the data to PCA
X_train_pca = pca.transform(X_train)                            # Transform
X_test_pca = pca.transform(X_test)
#print("Eigen Values: ", pca.explained_variance_)		        # Printing Eigen Values
#print("Eigen Vectors: ", pca.components_)
print('After Dimension Reduction:', X_train_pca.shape)          # After dimension reduction

# Please use training data set to build kernel Support vector classifier
from sklearn.svm import SVC
clf = SVC(kernel='linear')                                      # Applyng Support vector classifier
clf = clf.fit(X_train_pca,y_train)                              # Fit the training data to classifier
#print(clf.support_vectors_)

# Please use testing fata set to evaluate your model
y_pred = clf.predict(X_test_pca)                                # predict the classifier values
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred,target_names=faces.target_names)) # print classification report