#!/usr/bin/env python
# coding: utf-8

# In[248]:


import os, sys
import docx #need to first install python-docx module in Anaconda Prompt shell: "conda install -c conda-forge python-docx"
import pandas as pd
import numpy as np
import nltk
import scipy
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
import matplotlib as plt
import seaborn as sns


# In[249]:


path = 'C:/Users/ipersson/Job_Description'
list_folders = print(os.listdir(path))


# In[250]:


job_file = []
def list_files(path, extension):
    for (dirpath, dirname, filename) in os.walk(path):
        #job_file.append([file for file in filename if file.endswith(extension)])
        job_file.append([os.path.join(dirpath, file) for file in filename if file.endswith(extension)]) 


# In[251]:



list_files(path, '.docx')
# list_files(path, '.doc') 
# many of the document files are '*.doc' format instead of '*.docx', but docx module won't read '.doc' 

job_file = [file for sublist in job_file for file in sublist]


# In[252]:


print(job_file[:5])


# In[253]:


def readtxt(file):    
	doc = docx.Document(file)
	full_text = []
	for para in doc.paragraphs:
		full_text.append(para.text)    
	return '\n'.join(full_text)


# In[254]:


# extract job_text, job_class, and job_dept from files in job_descriptions list
# job_class is the unique 4-digit code given to each job description
# job_dept is one of 11 names given to folders that house job descriptions [i.e 'Administrative & Technical', Artistic & Creative', 'City Attorney', 'Council Staff', 'Emergency Communications', 'Exempt', 'Information Systems', 'Information Technology', 'Labor and Trade', 'Office Technical', 'Science & Technical']


# In[255]:


job_text = []
job_class = []
job_dept = []


# In[256]:


for file in job_file:
    job_text.append(readtxt(file))
    job_class.append(os.path.splitext(os.path.basename(file))[0])
    job_dept.append(file.split('\\')[-3])
       


# In[257]:


print (job_class[:1], job_dept[:1], job_text[:1])    


# In[258]:


job_tuple = list(zip(job_class, job_dept, job_file, job_text))
    


# In[259]:


print(job_tuple[:1])


# In[260]:


# create data frame with job_class, job_dept, and job_text as column variables

job_df= pd.DataFrame(job_tuple, columns=['job_class', 'job_dept', 'job_file', 'job_text'])
print(job_df.head())


# In[261]:


# Split data into train and test sets

x_train, x_test, y_train, y_test = train_test_split(job_df['job_text'],job_df['job_dept'],random_state=0)


# In[262]:


# vectorize with TF-IFD and analyze
# create a model based on Multinomial Naïve Bayes (NB)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# training the NB model with the train data

model.fit(x_train, y_train)

# created predicted job_dept values for the test data

y_predict = model.predict(x_test)

np.mean(y_predict == y_test)


# In[264]:


# vectorize with TF-IFD and analyze
# create a model based on K-Nearest Neighbors (KNN)

model = make_pipeline(TfidfVectorizer(), KNeighborsClassifier())

# training the KNN model with the train data

model.fit(x_train, y_train)

# created predicted job_dept values for the test data

y_predict = model.predict(x_test)

# compare predicted y values with actual y values
np.mean(y_predict == y_test)


# In[ ]:




