import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
from nltk.util import ngrams
from ngram import NGram
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

df =  pd.read_csv('/home/arpit/learning/machine learning/quora_dataset/test.csv')
a = set(df.question1.unique()) 
b = set(df.question2.unique())
unique_string = a.union(b)
print len(unique_string)
unique_string = list(unique_string)
data = {}
data['id']  = []
data['question'] = unique_string
for i in xrange(0,len(unique_string)):
	data['id'].append(i)
	# data['question'].append(unique_string[i])
df1 = pd.DataFrame(data=data)
print df1.head()
df1 = df1.dropna()
print df1.head()
df1.to_csv('question.csv',header=True, index=False)
print "saved"