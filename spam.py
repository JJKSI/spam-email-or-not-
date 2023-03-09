import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from numpy import dtype
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score
from sklearn.datasets import load_iris
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.preprocessing import LabelEncoder

#           DATA COLLECTION AND PREPROCESSING
df = pd.read_csv('mail_data.csv')

#print(df.isnull())

#replace null values with null string
df=df.where((pd.notnull(df)),'')# here we are replacing the null values with a null string

#print(df.head())

# to check number of row and col i.e size of the data
#df.shape()

# we are converting our category into numeric form( spam mail as 0 and ham mail as one)
le=LabelEncoder()

df.loc[df['Category'] == 'spam','Category',]=0# giving spam value as 0
df.loc[df['Category'] == 'ham','Category',]=1 # we didi similar thing here

#print(df.head())

# separating the data that is features and label

X=df['Message']
Y=df['Category']

#train test split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.30,random_state=4)

#feature extraction
# here we willl convert the message column into  numeric form using vectorization
#transformation of text into  vector form

tf=TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)#all words are checked and repeated words are checked and on basisi of it a new mail can be compared with mails that are spam and ham
#min df if score less than 1 than ignore the word or else vice versa
#stopwords=is so that was are etc they are not very use ful so it is good to ignore them
#lowercase=True means every letter is changed into lowercase while manupilating

#COVERTING into numeric values

xtrain1=tf.fit_transform(xtrain)# fit trasform is fitting the data in the ft function and will trasform data into numeric value
xtest1=tf.transform(xtest)#transform means we dont fit again we just need to get output here rather then fitting it.
#converting ytest and ytrain in to int form

#print(dtype(ytrain)) THE DATA TYPE of y set is object thats the reason why we need to convert it into int

ytrain=ytrain.astype('int')
ytest=ytest.astype('int')

# we can check xtrain and xtest values i.e in numeric form

#model selecting and training
lr=LogisticRegression()

lr.fit(xtrain1,ytrain)#( features and labels )

#evaluating the trained model

# prediction of training data
prediction=lr.predict(xtrain1)
acc=accuracy_score(ytrain,prediction)# true value and oredicted value

#print(prediction)
#print(acc)

# for test data???
# prediction for the test data
#
#BUILDING THE PREDICTION SYSTEM

input=['Having trouble deciding on a course or career? Try our free AECC Skills Personality Assessment to better understand your professional strengths, skills, and talents. It will only take a few minutes.']
# transforming the input data into numeric form
ip=tf.transform(input)

op=lr.predict(ip)
print(op)
# o means the mail is spam
# 1 means its ham







