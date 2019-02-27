import numpy as np
import pandas as pd
import xlsxwriter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
# -----------------------------------Data pre-process-------------------------------------------------------
# get  training data
# df = pd.read_excel('../data/test_customers.xlsx')
# df = pd.read_excel('daily_loads.xlsx')
df = pd.read_excel('average_loads.xlsx')
# df = pd.read_excel('../data/Customers_data.xlsx')
# df1 = pd.read_excel('../data/WPG_data_test.xlsx')
# -----------------------------------input data random pre-process------------------------------------
df = df.sample(frac=1)
# df_ran1 = df1.sample(frac=1)

# define which title to be noimal

# df_nominal = df["ID"]
df_nominal = df.index
# df_numerical_tmp1 = df1.ix[:, ['OH WK', 'OH FCST WK', 'BL WK', 'BL FCST WK', 'Last BL', 'Backlog', 'BL <= 9WKs', 'DC OH', 'On the way', 'Hub OH', 'Others OH', 'Avail.', 'Actual WK', 'FCST WK', 'Actual AWU', 'FCST AWU', 'FCST M', 'FCST M1', 'FCST M2', 'FCST M3']]
df_numerical_tmp = df.ix[:,:]
# df_numerical_tmp = df.ix[:, ['SEX','age','employment status','SOCIAL CLASS','people','people over 15','people under 15','accomodation']]
df_numerical = df_numerical_tmp.apply(pd.to_numeric, errors='coerce').fillna(-1)
# df_numerical1 = df_numerical_tmp1.apply(pd.to_numeric, errors='coerce').fillna(-1)



# get data dim to latter SOM prcess
input_dim = len(df_numerical.columns)
input_num = len(df_numerical.index)


# -----------------------------------input data random pre-process------------------------------------
# change data to np array (SOM accept nparray format)
input_data = np.array(df_numerical).astype(float)

scaler = StandardScaler().fit(input_data)
input_data = scaler.transform(input_data)

# search for an optimal value of K for KMeans

parameters={'n_clusters':range(1,28)}
km=KMeans(n_clusters=14)
# clf=GridSearchCV(km,parameters)
km.fit(X=input_data,y=None)
# print(sorted(clf.cv_results_.keys()))
# print(km.best_params_)
# print(clf.score(input_data))
results= [[]]*14
results=np.array(results)
positions = km.predict(input_data)
results={}
for index,i in enumerate(df_nominal):
    results[i]=positions[index]



print(results)