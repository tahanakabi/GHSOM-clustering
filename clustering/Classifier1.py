import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Classifier():
    def __init__(self, name, dataframe_number=1):
        #data frame number refers to the data frames defined hereunder df1, df2, df3

        df= pd.read_csv(name)
        self.file_name=df['Filename']
        df=df.replace('N',0)
        df=df.replace('Y',1)

        #adding the averages values to data frame
        df['Addition_C/W']=df['Addition_C']/df['W']
        df['Comparison_C/W']=df['Comparison_C']/df['W']
        df['Concession_C/W']=df['Concession_C']/df['W']
        df['Contrast_C/W']=df['Contrast_C']/df['W']
        df['Emphasis_C/W']=df['Emphasis_C']/df['W']
        df['Example_C/W']=df['Example_C']/df['W']
        df['Summary_C/W']=df['Summary_C']/df['W']
        df['Time_sequence_C/W']=df['Time_sequence_C']/df['W']
        df['Subject_P/W']=df['Subject_P']/df['W']
        df['Object_P/W']=df['Object_P']/df['W']
        df['Possessive_P/W']=df['Possessive_P']/df['W']
        df['Relative_P/W']=df['Relative_P']/df['W']
        df['Demonstrative_P/W']=df['Demonstrative_P']/df['W']
        df['Cause_C/W']=df['Cause_C']/df['W']

        # Define different dataframes for test
        # we just keep independent features due to the assumption of naive Bayes of features independence
        df0=df.drop(['Folder','#Par', 'Filename'],1)
        df1=df[['Paragraph','Fluent?','Con', 'Pron','W','S']]
        df2=df[['Paragraph','Fluent?','Addition_C','Comparison_C','Concession_C','Contrast_C',
                               'Emphasis_C','Example_C','Summary_C','Time_sequence_C','Subject_P',
                               'Object_P','Possessive_P','Relative_P','Demonstrative_P','Cause_C']]
        df3=df[['Paragraph','Fluent?','Average S.','Addition_C/W','Comparison_C/W','Concession_C/W','Contrast_C/W',
                                           'Emphasis_C/W','Example_C/W','Summary_C/W','Time_sequence_C/W','Subject_P/W',
                                           'Object_P/W','Possessive_P/W','Relative_P/W','Demonstrative_P/W','Cause_C/W']]
        df_list=[df0,df1,df2,df3]
        self.df=df_list[dataframe_number-1]
        #define several X and y according to data frames

        self.X=np.array(self.df.drop(['Fluent?'], 1))
        self.y=np.array(df['Fluent?'])
        self.features = self.df.drop(['Fluent?','Paragraph'],1).columns

    def split_data(self,X,y,test_size):
        # separate the data into training and testing
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=0)

        self.X_train_filename=self.X_train[:,0]
        self.X_train=self.X_train[:,1:]
        self.X_test_filename=self.X_test[:,0]
        self.X_test=self.X_test[:,1:]
        
    def bernouliNB(self):
    # BernoulliNB implements the naive Bayes training and classification algorithms for data that is distributed according
    # to multivariprint sorted(zip(map(lambda x: round(x, 4), clf.coef_[0]), features), reverse=True)ate Bernoulli distributions
        from sklearn.naive_bayes import BernoulliNB
        clf=BernoulliNB()
        clf.fit(self.X_train,self.y_train)
        return clf

    def multinomialNB(self):
        # MultinomialNB implements the naive Bayes algorithm for multinomially distributed data, and is one of the two
    # classic naive Bayes variants used in text classification (where the data are typically represented as word vector
    # counts, although tf-idf vectors are also known to work well in practice)
        from sklearn.naive_bayes import MultinomialNB
        clf=MultinomialNB()
        clf.fit(self.X_train,self.y_train)
        return clf

    def gaussianNB(self):
    # GaussianNB implements the Gaussian Naive Bayes algorithm for classification. The likelihood of the features
    # is assumed to be Gaussian
        from sklearn.naive_bayes import GaussianNB
        clf=GaussianNB()
        clf.fit(self.X_train,self.y_train)
        return clf

    def svm(self):
        from sklearn.svm import SVC
        clf = SVC(kernel='rbf', C=100, gamma=0.0001,probability=True)
        clf.fit(self.X_train, self.y_train)
        return clf

    def randomForest(self):
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_jobs=-1, n_estimators=10000, min_samples_leaf=50)
        clf.fit(self.X_train, self.y_train)
        return clf

    def kNN(self):
        from sklearn import neighbors
        clf=neighbors.KNeighborsClassifier(n_neighbors=35,weights='distance',n_jobs=-1)
        clf.fit(self.X_train,self.y_train)
        return clf

    def aNN(self):
        from sklearn.neural_network import MLPClassifier
        # For small datasets, however, 'lbfgs' can converge faster and perform better.
        clf=MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 3), random_state=1)
        clf.fit(self.X_train, self.y_train)
        return clf

    def results(self,clf,X_test, y_test,filename):
        # Determine probabilities of the classes for each data point
        df_proba = pd.DataFrame()
        prb_array = clf.predict_proba(X_test)
        for idx, proba in np.ndenumerate(prb_array):
            df_proba.loc[idx] = proba
        # Building the results matrix
        df_results = pd.DataFrame(index=np.array(filename))
        df_results['Probability of 0'] = np.array(df_proba[0])
        df_results['Probability of 1'] = np.array(df_proba[1])
        predicted_class = clf.predict(X_test)
        df_results['class'] = y_test
        df_results['predicted_class'] = predicted_class
        df_results['Correctly predicted?'] = (df_results['predicted_class'] == df_results['class'])
        # print df_results
        return df_results
        # saving results to excel
        # df_results.to_excel('BernoulliNB_results.xls')

    def determine_features_imoortance(self,clf):

    #Mean decrease impurity feature selection
    #The first featurs in the list should be the most important predictors
    #This is possible just for linear classifiers so it doesn't work with GaussianNB()
        print(sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), self.features), reverse=True))


# the number 3 refers to the third data frame defined before you can put 4,3,2,1 but don't use 4 for NB cause it contains dependent variables
nb_train=Classifier('new.csv', 4)
nb_train.split_data(nb_train.X,nb_train.y,0.2)

clfSVM=nb_train.svm()
print(clfSVM.score(nb_train.X_test, nb_train.y_test))
nb_train.results(clfSVM, nb_train.X_test, nb_train.y_test, nb_train.X_test_filename).to_excel('svmResults.xls')

clfRF=nb_train.randomForest()
print(clfRF.score(nb_train.X_test, nb_train.y_test))
nb_train.results(clfRF, nb_train.X_test, nb_train.y_test, nb_train.X_test_filename).to_excel('rfResults.xls')
nb_train.determine_features_imoortance(clfRF)

clf1=nb_train.bernouliNB()
print(clf1.score(nb_train.X_test, nb_train.y_test))
nb_train.results(clf1, nb_train.X_test, nb_train.y_test, nb_train.X_test_filename).to_excel('bernouliResults.xls')

clf2=nb_train.multinomialNB()
print(clf2.score(nb_train.X_test, nb_train.y_test))
nb_train.results(clf2, nb_train.X_test, nb_train.y_test, nb_train.X_test_filename).to_excel('multinomialResults.xls')

clf3=nb_train.gaussianNB()
print(clf3.score(nb_train.X_test, nb_train.y_test))
nb_train.results(clf3, nb_train.X_test, nb_train.y_test, nb_train.X_test_filename).to_excel('gaussianResults.xls')

clf4=nb_train.kNN()
nb_train.results(clf4, nb_train.X_test, nb_train.y_test, nb_train.X_test_filename).to_excel('knnResults.xls')
print(clf4.score(nb_train.X_test, nb_train.y_test))

clf5=nb_train.aNN()
print(clf5.score(nb_train.X_test, nb_train.y_test))
nb_train.results(clf5, nb_train.X_test, nb_train.y_test, nb_train.X_test_filename).to_excel('annResults.xls')

