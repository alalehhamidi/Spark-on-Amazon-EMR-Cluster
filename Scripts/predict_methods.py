
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

## Setting the parameter values
input_name = 'combined_csv.csv' #the name of the input file
output_name1 = 'RFoutput'
output_name2 = 'KNNoutput'
output_name3 = 'SVMoutput'
model1 = 'RF_model.pckl' #full address of the first model pickle file in S3
model2 = 'SVM_model.pckl'
model3 = 'knn_model.pckl'


##Get csv file from s3 and create a pandas.dataframe
import boto3
import io

REGION = 'us-east-1'
ACCESS_KEY_ID = '*****'
SECRET_ACCESS_KEY = '*****'

BUCKET_NAME = 'aws-logs-914250087788-us-east-1'
KEY = input_name # file path in S3

s3c = boto3.client(
        's3',
        region_name = REGION,
        aws_access_key_id = ACCESS_KEY_ID,
        aws_secret_access_key = SECRET_ACCESS_KEY
    )

"""
obj = s3c.get_object(Bucket= BUCKET_NAME , Key = KEY)
dataset = pd.read_csv(io.BytesIO(obj['Body'].read()), encoding='utf8')
"""
s3c.download_file(BUCKET_NAME, input_name, 'combined_csv.csv')
dataset = pd.read_csv('combined_csv.csv')


## Download pickles
rfModel=s3c.download_file(BUCKET_NAME, model1, 'RF_model.pckl')
svmModel=s3c.download_file(BUCKET_NAME, model2, 'SVM_model.pckl')
knnModel=s3c.download_file(BUCKET_NAME, model3, 'knn_model.pckl')


## Taking care of missing data (part1)
dataset = dataset[dataset['TEMP'].notna()]
#dataset = dataset[dataset['wd'].notna()]
"""
##Removing Outliers
q = dataset['CO'].quantile(0.98)
dataset = dataset[dataset['CO']<q]

q = dataset['NO2'].quantile(0.995)
dataset = dataset[dataset['NO2']<q]

q = dataset['PM10'].quantile(0.99)
dataset = dataset[dataset['PM10']<q]

q = dataset['WSPM'].quantile(0.99)
dataset = dataset[dataset['WSPM']<q]
"""

x = dataset[['month', 'day','hour', 'PM10', 'SO2', 'NO2', 'O3', 'DEWP', 'WSPM',  'CO']].values
y1 = dataset[['TEMP']].values
#y = dataset.iloc[:, 16]

## Taking care of missing data (part2)
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
missingvalues = missingvalues.fit(x[:, 3:10])
x[:, 3:10]=missingvalues.transform(x[:, 3:10])
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



##load model and predict

with open('RF_model.pckl', 'rb') as file:
    clf = pickle.load(file)

rf_pred=clf.predict(x)
np.savetxt(output_name1, rf_pred, delimiter=',', fmt='%s')

##Calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y, rf_pred)
print("Random Forest accuracy is:")
print(accuracy)




##load model and predict

with open('knn_model.pckl', 'rb') as file:
    clf = pickle.load(file)
knn_pred=clf.predict(x)
np.savetxt(output_name3, knn_pred, delimiter=',', fmt='%s')

##Calculate accuracy
accuracy = accuracy_score(y, knn_pred)
print("KNN accuracy is:")
print(accuracy)



## Drop 'CO' variable
x = x[:,0:9]

##load model and predict
with open('SVM_model.pckl', 'rb') as file:
    clf = pickle.load(file)
svm_pred=clf.predict(x)
np.savetxt(output_name2, svm_pred, delimiter=',', fmt='%s')

##Calculate accuracy
accuracy = accuracy_score(y, svm_pred)
print("SVM accuracy is:")
print(accuracy)




## Upload output_name file to S3
from botocore.exceptions import NoCredentialsError

s3c.upload_file(output_name1, BUCKET_NAME, output_name1)
s3c.upload_file(output_name2, BUCKET_NAME, output_name2)
s3c.upload_file(output_name3, BUCKET_NAME, output_name3)