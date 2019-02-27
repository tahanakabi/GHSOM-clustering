import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import xlsxwriter

df = pd.read_excel('load_shape_factors.xlsx')
X = np.array(df.ix[:,:-2]).astype(float)
line = np.array(df.ix[:,-2]).astype(float)
column = np.array(df.ix[:,-1]).astype(float)
clas =  np.array(df.ix[:,-1]).astype(float)

line_list=[]
column_list=[]
clas_list=[]
# for i,row in enumerate(X):
#     X_train = np.delete(X,i,0)
#     line_train= np.delete(line,i,0)
#     column_train = np.delete(column,i,0)
#     X_test = np.reshape(row,(1,row.shape[0]))
#     line_test = line[i]
#     column_test = column[i]
#
#     clf = SVC(kernel='poly', C=50,degree=5)
#     clf.fit(X_train, line_train)
#     line[i]=clf.predict(X_test)
#     line_list.append(line[i])
#
#     clf1 = SVC(kernel='poly', C=50,degree=5)
#     clf1.fit(X_train, column_train)
#     reclass = clf1.predict(X_test)
#     if column[i]!= reclass:
#         column[i]= column[i]+(reclass-column[i])//2
#     column_list.append(column[i])


for i,row in enumerate(X):
    X_train = np.delete(X,i,0)
    clas_train= np.delete(clas,i,0)

    X_test = np.reshape(row,(1,row.shape[0]))
    clas_test = clas[i]

    clf = SVC(kernel='rbf', C=1000)
    clf.fit(X_train, clas_train)
    new=clf.predict(X_test)
    # clas[i]=clf.predict(X_test)
    clas_list.append(new)



    # clf = RandomForestClassifier(n_jobs=-1, n_estimators=10000, min_samples_leaf=50)
    # clf.fit(self.X_train, self.y_train)

new_classes = np.array([clas_list])
print(new_classes)

workbook = xlsxwriter.Workbook("new_classes1.xlsx")
worksheet = workbook.add_worksheet()
for col, data in enumerate(new_classes):
    worksheet.write_column(0, col, data)
workbook.close()



