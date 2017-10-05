# Worked perfect except for democrat col
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df = pd.read_csv('house-votes-84.data', header=None)
# print(df)
print('No of records , attributes:', df.shape)                                           # (435, 17)
print(df.columns)
print(type(df))

# Replacing Y with 1 and N with 0
df = df.replace('y', 1)
df = df.replace('n', 0)
df = df.replace('republican', 1)
df = df.replace('democrat', 0)
#print('After replacement:', df)

df = df.replace('?', np.NaN)                            # Replace missing value with NaN
print('NaN replaced data set: ', df)
#print(df.isnull().head())

imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imp.fit(df,(435,17))
df_clean = imp.transform(df)
print('Clean Data Set:', df_clean)
print(df_clean.shape)
print(type(df_clean))

column_list = ['republican','handicapped-infants','water-project-cost-sharing','adoptionof-the-budget-resolution',
               'physician-fee-freeze','el-salvador-aid','eligious-groups-inschools','anti-satellite-test-ban'
    ,'aid-to-nicaraguan-contras','mx-missile','immigration','synfuelscorporation-cutback',
    'education-spending','superfund-right-to-sue','crime','dutyfree-exports','export-administration-act-south-africa']

df1 = pd.DataFrame(np.array(df_clean), columns=column_list)
df1 = df1.astype(int)
print('New data set with column names:', df1)

df1['Democrat'] = df1.republican.apply(lambda x: 0 if x == 1 else 1)
print('New Democrat column:', df1['Democrat'])
freq_itemsets=apriori(df1,min_support=0.3,use_colnames=True)
print(freq_itemsets)

df2 = pd.DataFrame(data=freq_itemsets)
ass_rules = association_rules(df2,metric="confidence", min_threshold=0.9)
print(ass_rules)