#!/usr/bin/env python
# coding: utf-8

# In[26]:


import os, sys
import re
from tika import parser
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
import scipy

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

import collections
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')






# In[18]:


path = r'C:\Users\ipersson\Job_Description'
list_folders = print(os.listdir(path))


# In[19]:


job_filepath = []
def list_files(path, extension):
    for (dirpath, dirname, filename) in os.walk(path):
        job_filepath.append([os.path.join(dirpath, file) for file in filename if file.endswith(extension)]) 


# In[20]:


list_files(path, '.pdf')
# list_files(path, '.doc') - throws an error message ?
# many of the document files are '*.doc' format instead of '*.docx', but docx module won't read '.doc' 

job_filepath = [file for sublist in job_filepath for file in sublist]


# In[21]:


def readtxt(file):    
	doc = parser.from_file(file)['content'] 
	return doc


# In[22]:


job_text = []
job_filename = []
job_dept = []


# In[23]:


for file in job_filepath:
    try:
        job_text.append(readtxt(file))
        job_filename.append(os.path.splitext(os.path.basename(file))[0])
        job_dept.append(file.split('\\')[-3])
    except Exception as e:
        print(str(e))


# In[27]:


# extract job codes, series, groups, etc. 

# regex patterns
p_class = re.compile('.*Class Code\:\s*(.*?)\\n')
p_eeo = re.compile('.*EEO Code\:\s*(.*?)\\n')
p_pay = re.compile('.*Pay Code\:\s*(.*?)\\n')
p_group = re.compile('.*Group\:\s*(.*?)\\n')
p_series = re.compile('.*Series\:\s*(.*?)\\n')
p_date = re.compile('.*Effective Date\:\s*(.*)') 


# In[28]:


def extract_values(text, pattern):
    return [match.group(1) for string in text for match in [pattern.search(string)] if match]

job_class = extract_values(job_text, p_class)
job_eeo = extract_values(job_text, p_eeo)
job_pay = extract_values(job_text, p_pay)
job_group = extract_values(job_text, p_group)
job_series = extract_values(job_text, p_series)
job_date = extract_values(job_text, p_date)

print(job_class[:1], job_eeo[:1], job_pay[:1], job_group[:1], job_series[:1], job_date[:1])


# In[29]:


# check if filename matches class code

job_class.sort()
job_filename.sort()

if job_class == job_filename:
    print('same')
else: 
    print('not same')
    print('unique filenames:', np.setdiff1d(job_filename, job_class, assume_unique=True))
    print('unique class names:', np.setdiff1d(job_class, job_filename, assume_unique=True))


# In[32]:


job_tuple = list(zip(job_filename, job_class, job_series, job_group, job_dept, job_eeo, job_pay, job_date, job_filepath, job_text))


# In[34]:


# create data frame with job_class, job_dept, and job_text as column variables

job_df= pd.DataFrame(job_tuple, columns=['job_filename', 'job_class', 'job_series', 'job_group', 'job_dept', 'job_eeo', 'job_pay', 'job_date', 'job_file', 'job_text'])


# In[35]:


print(job_df.info())
print(job_df.describe())


# In[37]:


print(job_df.head())


# In[38]:


print(str(len(job_df['job_dept'].unique())) + ' unique departments:', job_df['job_dept'].unique())
print(str(len(job_df['job_group'].unique())) + ' unique groups:', job_df['job_group'].unique())
print(str(len(job_df['job_series'].unique())) + ' unique series:', job_df['job_series'].unique())
print(str(len(job_df['job_pay'].unique())) + ' unique pay codes:', job_df['job_pay'].unique())


# In[2]:


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# In[289]:


le = LabelEncoder()
Y = le.fit_transform(job_df['job_dept'])
x_train, x_test, y_train, y_test = train_test_split(job_df['job_text'],job_df['job_dept'],random_state=0,test_size = 0.20)


# In[269]:


collections.Counter(y_train)


# In[270]:


collections.Counter(y_test)


# In[271]:


stop_words = stopwords.words('english')


# In[272]:


model = make_pipeline(TfidfVectorizer(stop_words=stop_words), MultinomialNB())
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
np.mean(y_predict == y_test)


# In[274]:


class_names = ['Labor and Trade',
         'Exempt',
         'Administrative & Technical',
         'Science & Technical',
         'Information Systems',
         'Office Technical',
         'Information Technology',
         'City Attorney',
         'Council Staff',
         'Emergency Communications',
         'Artistic & Creative']


# In[275]:


np.set_printoptions(precision=2)
plot_confusion_matrix(y_test, y_predict, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# In[276]:


model = make_pipeline(TfidfVectorizer(stop_words=stop_words), KNeighborsClassifier())

# training the KNN model with the train data

model.fit(x_train, y_train)

# created predicted job_dept values for the test data

y_predict = model.predict(x_test)

# compare predicted y values with actual y values
np.mean(y_predict == y_test)


# In[277]:


plot_confusion_matrix(y_test, y_predict, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# In[290]:





# In[291]:


le = LabelEncoder()
Y = le.fit_transform(job_df['job_dept'])
x_train, x_test, y_train, y_test = train_test_split(job_df['job_text'],Y,random_state=0,test_size = 0.20)


# In[293]:


max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(x_train)
sequences = tok.texts_to_sequences(x_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)


# In[369]:


def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(11,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model


# In[407]:


model = RNN()
model.summary()
model.compile(loss='sparse_categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])


# In[424]:


model.fit(sequences_matrix,y_train,batch_size=50,epochs=1000,
          validation_split=0.2)


# In[427]:


test_sequences = tok.texts_to_sequences(x_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)


# In[428]:


accr = model.evaluate(test_sequences_matrix,y_test)


# In[429]:


print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


# In[430]:


y_predict = model.predict(test_sequences_matrix)


# In[431]:


y_predict = np.argmax(y_predict,axis=1)


# In[432]:


plot_confusion_matrix(y_test, y_predict, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# In[ ]:




