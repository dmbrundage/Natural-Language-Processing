# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 17:35:56 2018

@author: dmbru
"""

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import nltk
import gzip
import gensim
from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec
LabeledSentence = gensim.models.doc2vec.LabeledSentence
import logging
from string import punctuation
import string
import re
from tqdm import tqdm

import pymysql.cursors  

 ###Data Loading###
# Connect to the database.
user = 'userid.txt'
password = 'password.txt'
connection = pymysql.connect(host='localhost',
                             user = user,
                             password= password,                             
                             db='mimic',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)
 
print ("connect successful!!")
 

# SQL Queries

nsql = "Select *  from mimic.noteevents where noteevents.category = 'Nursing' limit 500"
dsql = "Select *  from mimic.noteevents where noteevents.category = 'General' limit 400"
#225000

nursing = pd.read_sql(nsql, connection)
general = pd.read_sql(dsql, connection)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

###Create Artifical Dataset
df = nursing['TEXT']
df = pd.DataFrame(df)
gf = pd.DataFrame(general['TEXT'])
df['TEXT_B']=df['TEXT']

df['DUP'] = 1
df['DUP'][100:499] = 0
x=100
i=0
while x <= 500:
    df['TEXT_B'][x] = gf['TEXT'][i]
    print(df['TEXT_B'][x] == df['TEXT'][i])
    i=i+1
    x=x+1
i=0 
for text in df['TEXT']:
    if text == df['TEXT_B'][i]:
        print('True')
        df['DUP'][i] = 1
        i=i+1
    else:
        print('False')
        df['DUP'][i] = 0
        i=i+1
        
        

###Text Preprocessing
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
import nltk
sno = nltk.stem.SnowballStemmer('english')
#Toeknize sentences
from nltk.tokenize import sent_tokenize
df['sent_tokens_a'] = df.apply(lambda row: sent_tokenize(row['TEXT']), axis=1)
df['sent_tokens_b'] = df.apply(lambda row: sent_tokenize(row['TEXT_B']), axis=1)

#Tokenize words
from nltk.tokenize import word_tokenize
df['word_tokens_a'] = df.apply(lambda row: word_tokenize(row['TEXT']), axis=1)
df['word_tokens_b'] = df.apply(lambda row: word_tokenize(row['TEXT_B']), axis=1)

#Stemmed Tokens
df['stemmed_a'] = df["word_tokens_a"].apply(lambda x: [sno.stem(y) for y in x])
df['stemmed_b'] = df["word_tokens_b"].apply(lambda x: [sno.stem(y) for y in x])

##Feature Building

def symm_similarity(textA,textB):
    textA = set(word_tokenize(textA))
    textB = set(word_tokenize(textB))    
    intersection = len(textA.intersection(textB))
    difference = len(textA.symmetric_difference(textB))
    return intersection/float(intersection+difference) 

def jaccard_similarity(query, document):
    w1 = set(query)
    w2 = set(document)
    return nltk.jaccard_distance(w1,w2)

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
def get_cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]


df['jaccard_distance'] = df.apply(lambda row: jaccard_similarity(row['TEXT'], row['TEXT_B']), axis=1)

tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(df['TEXT'])
df['tfidf_a'] =  df.apply(lambda row: tfidf_vect.transform(row['word_tokens_a']), axis=1)
df['tfidf_b'] =  df.apply(lambda row: tfidf_vect.transform(row['word_tokens_b']), axis=1)

df['cosine_sim'] = df.apply(lambda row: get_cosine_sim(row['TEXT'], row['TEXT_B']), axis=1)

df['sent_a_len'] = df.apply(lambda row: len(row['sent_tokens_a']), axis=1)
df['sent_b_len'] = df.apply(lambda row: len(row['sent_tokens_b']), axis=1)

df['sent_diff'] = df.apply(lambda row: abs(row['sent_a_len']-row['sent_b_len']), axis=1)

df['shared_token_precent'] = df.apply(lambda row: symm_similarity(row['TEXT'], row['TEXT_B']), axis=1)


##Predictive Model
X = df
y = df['DUP']

X = X.drop(['DUP','TEXT','TEXT_B','clean_text_a','word_tokens_a','word_tokens_b','stemmed_a','stemmed_b','sent_tokens_a','sent_tokens_b','clean_text_b'], axis=1)
X = X.drop(['tfidf_a','tfidf_b'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=123, stratify=y)

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, y_test)
    

# SVM on Ngram Level TF IDF Vectors
accuracy = train_model(svm.SVC(), X_train, y_train, X_test)
print ("SVM, N-Gram Vectors: ", accuracy)
