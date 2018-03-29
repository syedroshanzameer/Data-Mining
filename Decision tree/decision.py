"""
Name : Roshan Zameer Syed
ID: 99999-2920
Project 5 : Decision tree
"""
import pandas as pd
import numpy as np

df = pd.read_csv('breast-cancer.data',header=None)  #Load breast cancer data set
#print(df)
print(df.shape)

#print(df[10])
df_class = df[10]                                   #Assigning last column to class
df = df.drop(df.columns[10],axis=1)                 # dropping the last column
#print(df)
print(df.shape)
df = df.replace('?',np.NaN)                         #replacing missing values with NaN

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)     #Imputing the values
imp.fit(df)
df_clean = imp.transform(df)                                            #transform
#print(df_clean.shape)

column_list = ['Sample Code no.','Clump Thickness','Uniformity of cell size','Uniformity of cell shape',
               'Marginal adhesion','Single Epithelial cell size','Bare Nuclei','Bland Chromatin',
               'Normal Nucleoli','Mitoses']                             #creating a header list
df_clean = pd.DataFrame(np.array(df_clean),columns=column_list)
df_clean = df_clean.astype(int)
#print(df_clean)


X = df_clean
y = df_class

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 1)

from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_predict))

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,y_predict))


tree.export_graphviz(model.tree_, out_file='tree.dot', feature_names=X.columns)

from subprocess import call
call(['dot','-T','png','tree.dot','-o','tree.png'])

#call(['dot tree.dot -Tpng -o tree1.png'])
# $ dot foo.dot -Tpng -o foo.png