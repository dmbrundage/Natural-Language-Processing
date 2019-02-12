# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import sklearn
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

user='user'
password='password'
connection = pymysql.connect(host='localhost',
                             user=user,
                             password=password,                             
                             db='mimic',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)
 
print ("connect successful!!")
 

# SQL Queries
rsql = "Select *  from mimic.noteevents where noteevents.category = 'Radiology' limit 1000"
nsql = "Select *  from mimic.noteevents where noteevents.category = 'Nursing' limit 1000"
psql = "Select *  from mimic.noteevents where noteevents.category = 'Physician' limit 1000"
ssql = "Select *  from mimic.noteevents where noteevents.category = 'Social Work' limit 1000"

testsql = "Select *  from mimic.noteevents where noteevents.category = 'Physician' limit 100000"

radiology = pd.read_sql(rsql, connection)
nursing = pd.read_sql(nsql, connection)
physician = pd.read_sql(psql, connection)
socialwork = pd.read_sql(ssql, connection)


radiology = radiology.reset_index(drop=True)
nursing = nursing.reset_index(drop=True)
physician = physician.reset_index(drop=True)
socialwork = socialwork.reset_index(drop=True)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 
frames = [radiology, nursing, physician, socialwork]


#Load note data and create corpus
data = pd.concat(frames)
DF = pd.DataFrame(data)
text = DF['TEXT']
corpus = text.str.cat(sep=' ')
#remove deidentification patterns
data['clean_text'] = data.apply(lambda row: re.sub('\[\*\*(\w*\s*|\(\w*\)|(\d*\-))*\*\*\]', ' ', row['TEXT']), axis=1)
data['clean_text'] = data['clean_text'].str.replace('[^\w\s]','')


#Stemming
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
stem = porter_stemmer.stem(corpus)
data['clean_text'] = data['clean_text'].apply(porter_stemmer.stem)

#Toeknize sentences
from nltk.tokenize import sent_tokenize
sent_tokenize_list = sent_tokenize(corpus)
data['sent_tokens'] = data['clean_text'].apply(sent_tokenize)


#Tokenize words
from nltk.tokenize import word_tokenize
word_tokenize_list = word_tokenize(stem)
data['word_tokens'] = data['clean_text'].apply(word_tokenize)

#remove punctuation
cleanword = re.sub(r'[^\w\s]','',stem)
lowercased_sents = [sent.lower() for sent in sent_tokenize_list]
discard_punctuation_and_lowercased_sents = [re.sub(r'[^\w\s]','',sent) for sent in lowercased_sents]

#Labelize data
def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in tqdm(enumerate(tweets)):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

testing = labelizeTweets(discard_punctuation_and_lowercased_sents, 'training')
sent1 = [word_tokenize(sent) for sent in discard_punctuation_and_lowercased_sents]




from nltk.corpus import stopwords
#filter stop words
filtered_words = [word for word in sent1 if word not in stopwords.words('english')]

#Word2vec Model
medrec_w2v = Word2Vec(size=200, min_count=10, iter=1000)
medrec_w2v.build_vocab(filtered_words)
medrec_w2v.train(filtered_words, epochs=medrec_w2v.iter, total_examples=medrec_w2v.corpus_count)
print(medrec_w2v.most_similar(["history"]))

#Authorship Identification
author = data[['clean_text','CATEGORY']]
train_X, test_X, train_y, test_y = model_selection.train_test_split(author['clean_text'], author['CATEGORY'], random_state =42)



###TF-IDF Feature Engineering)
# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(author['clean_text'])
xtrain_tfidf =  tfidf_vect.transform(train_X)
xvalid_tfidf =  tfidf_vect.transform(test_X)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(author['clean_text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_X)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(test_X)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(author['clean_text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_X) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(test_X) 



###Document Classification###
#train model pipeline
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, test_y)
    

# SVM on Ngram Level TF IDF Vectors
accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print ("SVM, N-Gram Vectors: ", accuracy)

# Naive Bayes on Word Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print ("NB, WordLevel TF-IDF: ", accuracy)

# Naive Bayes on Ngram Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print ("NB, N-Gram Vectors: ", accuracy)

# Naive Bayes on Character Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print ("NB, CharLevel Vectors: ", accuracy)

# Linear Classifier on Word Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
print ("LR, WordLevel TF-IDF: ", accuracy)

# Linear Classifier on Ngram Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print ("LR, N-Gram Vectors: ", accuracy)

# Linear Classifier on Character Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print ("LR, CharLevel Vectors: ", accuracy)

# RF on Word Level TF IDF Vectors
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
print ("RF, WordLevel TF-IDF: ", accuracy)



#Topic Modeling

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt


# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# Convert to list
topicdata = data.clean_text.values.tolist()

# Remove new line characters
topicdata = [re.sub('\s+', ' ', sent) for sent in topicdata]

# Remove distracting single quotes
topicdata = [re.sub("\'", "", sent) for sent in topicdata]

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(topicdata))

print(data_words[:1])

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[0]]])
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Remove Stop Words
import en_core_web_lg
nlp = en_core_web_lg.load()
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1])

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

# Print the Keyword in the 10 topics
print(lda_model.print_topics())
doc_lda = lda_model[corpus]


# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis
