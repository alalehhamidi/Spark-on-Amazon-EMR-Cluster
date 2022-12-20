import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the dataset
dataset = pd.read_csv('test3.csv')


## Taking care of missing data (part1)
dataset = dataset[dataset['TEMP'].notna()]
#dataset = dataset[dataset['wd'].notna()]

##Removing Outliers
q = dataset['CO'].quantile(0.98)
dataset = dataset[dataset['CO']<q]

q = dataset['NO2'].quantile(0.995)
dataset = dataset[dataset['NO2']<q]

q = dataset['PM10'].quantile(0.99)
dataset = dataset[dataset['PM10']<q]

q = dataset['WSPM'].quantile(0.99)
dataset = dataset[dataset['WSPM']<q]


x = dataset[['month', 'day','hour', 'PM10', 'SO2', 'NO2', 'O3', 'DEWP', 'WSPM']].values
y1 = dataset[['TEMP']].values
#y = dataset.iloc[:, 16]

## Taking care of missing data (part2)
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
missingvalues = missingvalues.fit(x[:, 3:9])
x[:, 3:9]=missingvalues.transform(x[:, 3:9])
"""
## Encoding categorical data (station & wd)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

ct1 = ColumnTransformer([('encoder', OneHotEncoder(), [10])], remainder='passthrough')
x = ct1.fit_transform(x)
#To avoid the Dummy variable trap:
x = x[:,1:]

ct2 = ColumnTransformer([('encoder', OneHotEncoder(), [23])], remainder='passthrough')
#x = ct2.fit_transform(x)
x = np.array(ct2.fit_transform(x), dtype=np.float)
#To avoid the Dummy variable trap:
x = x[:,1:]
"""
##Changing Y values to the categories

y=dataset[['wd']].values
for i in range(len(y1)):

        if y1[i]<0:
            y[i]='verycold'
        elif (y1[i]>=0) and (y1[i]<10):
            y[i]='cold'
        elif (y1[i] >= 10) and (y1[i] < 20):
            y[i] = 'moderate'
        elif (y1[i] >= 20) and (y1[i] < 30):
            y[i] = 'hot'
        elif y1[i]>=30:
            y[i] = 'veryhot'



## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
"""
## Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
X_test = sc_x.transform(x_test)
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)
"""
## Fitting Classifier to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state = 0)
classifier.fit(x_train,np.ravel(y_train,order='C'))


##Predicting the test set result
y_pred = classifier.predict(x_test)
"""
##Making Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
"""


##Calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

##Save Model
import pickle

f = open('SVM_model.pckl' , 'wb')
pickle.dump(classifier, f)
f.close()

