from sklearn import datasets
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

import numpy as np
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris.data,iris.target)

y_pred = gnb.fit(X_train,y_train).predict(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


print (X_train.shape)
print(X_test.shape) 
print(confusion_matrix(y_pred,y_test))
print(f1_score(y_pred,y_test,average=None))


