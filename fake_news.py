import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')
news_data = pd.read_csv('news.csv', engine='python', on_bad_lines='skip')
news_data.shape
news_data.head()
news_data.isnull().sum()
news_data=news_data.fillna('')
news_data['content']=news_data['author']+' '+news_data['title']
X=news_data.drop(columns='label',axis=1)
Y=news_data['label']
#Stemming
port_stem=PorterStemmer()
news_data['content']=news_data['content'].apply(stemming)
X=news_data['content'].values
Y=news_data['label'].values
#text to numbers
vectorizer=TfidfVectorizer()
vectorizer.fit(X)
X=vectorizer.transform(X)
print(X.shape)
print(Y.shape)
Y = pd.Series(Y)
print(Y.head())
print(Y.unique())
most_frequent = Y.value_counts().idxmax()
print(type(Y[0]))  # Check the type of the first element
print(pd.Series(Y).apply(type).value_counts())  # Count different types in Y
Y = pd.to_numeric(Y, errors='coerce')
most_frequent = Y.value_counts().idxmax()
Y = Y.fillna(most_frequent)
print(Y.unique())  # Should show only valid numeric labels
print(Y.dtype)     # Should be float64 or int64
X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)
model=LogisticRegression()
model.fit(X_train,Y_train)
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print(training_data_accuracy)
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print(test_data_accuracy)
#Predictive System
X_new=X_test[4] #Checking the 5th data of the test set
prediction=model.predict(X_new)
print(prediction)
if(prediction[0]==0):
  print("This is a Real News")
else:
  print("This is a Fake News")
print(f"Actual Label: {news_data['label'][4]}")  # Shows actual label for that article
