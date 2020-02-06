
# coding: utf-8

# In[203]:


#Importing the libraries
import nltk
from nltk.corpus import stopwords
from textblob import Word
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[158]:


#Importing the train and test Datasets
train_dataset=pd.read_csv(r'C:\Users\Sonu\Downloads\datasetc062cf9\dataset\train.csv',encoding='ISO-8859-1')
test_dataset=pd.read_csv(r'C:\Users\Sonu\Downloads\datasetc062cf9\dataset\test.csv',encoding= 'ISO-8859-1')


# In[159]:


#Taking a look at the number of rows and columns of the datasets
train_dataset.shape


# In[160]:


test_dataset.shape


# In[161]:


#Exploring the column namespresent in the datasets
train_dataset.columns


# In[162]:


test_dataset.columns


# In[163]:


test_dataset[['Index','Unnamed: 9']].head()


# In[164]:


#Dropping the not so required columns 'Index' and 'Unnamed: 9' as they are not giving us any specific relevant information
test_dataset.drop(columns= ['Index','Unnamed: 9'],inplace=True)


# In[165]:


test_dataset.columns


# In[166]:


train_dataset.info()


# In[167]:


train_dataset.isnull().sum()


# In[168]:


#For the single missing value in TRANS_CONV_TEXT, we apply imputation by mode to fill the value in
train_dataset['TRANS_CONV_TEXT'] = train_dataset['TRANS_CONV_TEXT'].fillna(train_dataset['TRANS_CONV_TEXT'].mode()[0])


# In[169]:


#Verifying no missing values in TRANS_CONV_TEXT
train_dataset.isnull().sum()


# In[170]:


#Extracting the values of  TRANS_CONV_TEXT and Patient_Tag and separating them out in a Dataframe
New_train= train_dataset[['TRANS_CONV_TEXT','Patient_Tag']]
New_test= test_dataset[['TRANS_CONV_TEXT']]


# In[171]:


New_train.shape


# In[172]:


New_test.shape


# In[173]:


New_train['TRANS_CONV_TEXT'].apply(len).describe()


# In[174]:


New_test['TRANS_CONV_TEXT'].apply(len).describe()


# In[175]:


#Exploring distribution of Patient Tag
New_train['Patient_Tag'].value_counts()


# In[217]:


#Splitting the train into another train and validation sets in 70:30 ratio respectively

X_train, X_valid, y_train, y_valid = train_test_split(New_train['TRANS_CONV_TEXT'], New_train['Patient_Tag'],                                                     test_size=0.3, random_state=42)


# In[218]:


#Converting to lower case

New_train['TRANS_CONV_TEXT']=New_train['TRANS_CONV_TEXT'].str.lower()


# In[178]:


#Lemmatization

New_train['TRANS_CONV_TEXT']=New_train['TRANS_CONV_TEXT'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()] ))


# In[180]:


#Removal of digits

from string import digits

def remove_digits(s: str) -> str:
    remove_digits = str.maketrans('', '', digits)
    res = s.translate(remove_digits)
    return res


# In[181]:


X_train = X_train.apply(remove_digits)


# In[220]:


#Applying count vectorizer on the text

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words=None,
                             ngram_range=(1, 1), min_df=2, max_df=0.4, binary=True)

train_features = vectorizer.fit_transform(X_train)
train_labels = y_train

valid_features = vectorizer.transform(X_valid)
valid_labels = y_valid


# In[188]:


#Applying Logistic regression
model = LogisticRegression()
model.fit(train_features, train_labels)

valid_preds = model.predict(valid_features)
print(classification_report(valid_labels, valid_preds))
print(f'Accuracy:{accuracy_score(valid_labels, valid_preds)}')


# In[202]:


#Applying XGBoost
model = XGBClassifier()
model.fit(train_features, train_labels)

valid_preds = model.predict(valid_features)
print(classification_report(valid_labels, valid_preds))
print(f'Accuracy:{accuracy_score(valid_labels, valid_preds)}')


# In[219]:


#Applying Bernoulli's Naive Bayes
model = BernoulliNB(fit_prior=True)
model.fit(train_features, train_labels)

valid_preds = model.predict(valid_features)
print(classification_report(valid_labels, valid_preds))
print(f'Accuracy:{accuracy_score(valid_labels, valid_preds)}')


# In[204]:


#Applying Random Forest Classifier
model = RandomForestClassifier()
model.fit(train_features, train_labels)

valid_preds = model.predict(valid_features)
print(classification_report(valid_labels, valid_preds))
print(f'Accuracy:{accuracy_score(valid_labels, valid_preds)}')


# In[206]:


#To make predictions and prepare the submissions file 
#Repeating the pre processing steps on the test dataset
New_test['TRANS_CONV_TEXT'] = New_test['TRANS_CONV_TEXT'].apply(remove_digits)

New_test['TRANS_CONV_TEXT'] = New_test['TRANS_CONV_TEXT'].str.lower()

New_test['TRANS_CONV_TEXT'] = New_test['TRANS_CONV_TEXT'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()] ))


# In[208]:


test_features = vectorizer.transform(New_test['TRANS_CONV_TEXT'])


# In[209]:


test_preds = model.predict(test_features)


# In[211]:


test_for_submission = pd.read_csv(r'C:\Users\Sonu\Downloads\datasetc062cf9\dataset\test.csv',encoding= 'ISO-8859-1')


# In[212]:


submission = pd.DataFrame()
submission['Index'] = test_for_submission['Index']
submission['Patient_Tag'] = test_preds

submission.to_csv('submission.csv',index=False)


# In[222]:


submission.head(15)

