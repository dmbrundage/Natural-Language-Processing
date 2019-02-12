# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 18:23:16 2019

@author: dmbru
"""
from __future__ import print_function
import nltk
import pandas as pd
import re
import os
import codecs
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import matplotlib as plt
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import ward, dendrogram
from nltk.stem.snowball import SnowballStemmer

#load stopwords and stemmer
stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer('english')

#define tokenizer and stemmer
def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenize_only(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

#load text to cluster
df = pd.read_csv('text.txt', sep=',')
feedback = df['Target Text'].tolist()
totalvocab_stemmed = []
totalvocab_tokenized = []

for i in df['Target Text']:
    allwords_stemmed = tokenize_and_stem(i)
    totalvocab_stemmed.extend(allwords_stemmed)
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)
    
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_features=500, min_df=0.5, max_df=0.80,
                                   stop_words='english', use_idf=True,
                                   tokenizer=tokenize_and_stem, ngram_range=(1,3), norm = 'l1')
tfidf_matrix = tfidf_vectorizer.fit_transform(feedback)
print(tfidf_matrix.shape)
terms = tfidf_vectorizer.get_feature_names()
dist = 1 - cosine_similarity(tfidf_matrix)

#clustering and determining optimal K
sum_of_squared_distance = []
num_cluster = range(1, 25)
for k in num_cluster:
    km = KMeans(n_clusters=k)
    km.fit(tfidf_matrix)
    sum_of_squared_distance.append(km.inertia_)

#Plot Elbow Curve
plt.plot(num_cluster, sum_of_squared_distance, 'bx-')
plt.xlabel('K')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow optimal k')
plt.show()

#fit on optimal K 
#    km = KMeans(n_clusters=k)
#    km.fit(tfidf_matrix)


#pickle and load pickle
joblib.load('feedback_cluster.pkl)
km = joblib.load('feedback_cluster.pkl')
clusters = km.labels_.tolist()



feedback_dict = {'feedback':feedback,'cluster':clusters}
feedback_df = pd.DataFrame(feedback_dict, index = [clusters], columns = ['feedback','cluster'])
feedback_df['cluster'].value_counts()

print("Top terms per cluster:")
print()

#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
cluster_names = {}
cluster_name_list = []

for i in range(num_clusters):
    key = i
    print("Cluster %d words:" % i, end='')

#replace 3 with n words per cluster   
    for ind in order_centroids[i, :5]:
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
        if key in cluster_names:
            cluster_names[i].append(vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'))
        else:
            cluster_names[i] = [vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore')]
    print()
    print()
    print("Cluster %d feedback:" % i, end='')
    for feedback in feedback_df.loc[[i],['feedback']].values.tolist():
        print(' %s,' % feedback, end='')
    print()
    print()
print()
print()

#Visualize Clusters
cluster_colors ={0:'#1b9e77',1:'#d95f02',2:'#7570b3',3:'#e7298a',4:'#66a61e'}
MDS()
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)
xs, ys = pos[:, 0], pos[:, 1]
#Create data frame and group by cluster
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=feedback))
groups = df.groupby('label') 
#set up plot
fig, ax = plt.subplots(figsize=(17,9))
ax.margins(0.05)
for name, group in groups:
     ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
             label=cluster_names[name], color=cluster_colors[name],
                               mec='none')
     ax.set_aspect('auto')
     ax.tick_params(\
                   axis='x',
                   which='both',
                   bottom='off',
                   top='off',
                   labelbottom='off',
                   labeltop='off',
                   labelleft='off',
                   labelright='off')
     ax.tick_params(\
                   axis='y',
                   which='both',
                   left='off',
                   top='off',
                   labelbottom='off',
                   labeltop='off',
                   labelleft='off',
                   labelright='off')
ax.legend(numpoints=1)

plt.show()
plt.savefig('clusters_small_noaxes.png', dpi=200)

#Dendrogram
linkage_matrix = ward(dist)
fig, ax = plt.subplots(figsize=(15,20))
ax = dendrogram(linkage_matrix, orientation="right", labels=feedback)
plt.tick_params(\
                axis='x',
                which='both',
                bottom='off',
                top='off',
                labelbottom='off')
plt.tight_layout()
plt.savefig('ward_clusters.png', dpi=200)
