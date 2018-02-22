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
import nltk
from nltk.stem import WordNetLemmatizer
import sys 
import math
from collections import Counter 
import string
from scipy.stats import mode

reload(sys)  
sys.setdefaultencoding('utf8')

lemmatizer = WordNetLemmatizer()
stops = set(stopwords.words("english"))                 


# def save_csv(data):
# 	df1 = pd.read_csv('result1.csv')
# 	df2 = pd.DataFrame(data=data,columns=['test_id','is_duplicate'])
# 	frame = [df1,df2]
# 	df = pd.concat(frame)
# 	df.to_csv('result1.csv',header=True, index=False)
# 	print "saved"

# def create_csv():
# 	df = pd.DataFrame(columns=['test_id','is_duplicate'])
# 	df.to_csv('result1.csv',header=True, index=False)
# 	print "csv created"

def tokernize_removestop(a):
	a = a.lower()
	# print a
	a = re.sub('[():?.,]',"",a)
	# print a
	a = word_tokenize(a)
	c = a
	a = [w for w in a if not w in stops]
	if(len(a)==0):
		a=c
	c = []
	for w in a:
		try:
			c.append(lemmatizer.lemmatize(w))
		except:
			c.append(w)
	a =  c
	# print "token",a
	return a
	
def add_pos_tag(a):
	return nltk.pos_tag(a)

def jaccard_distance(string,n):
	# print a, b
	for i in xrange(0,n):
		for j in xrange(i+1,n):
			a = string[i].decode('utf-8')
			b = string[j].decode('utf-8')
			a = tokernize_removestop(a)
			b = tokernize_removestop(b)
			# a = add_pos_tag(a)
			# b = add_pos_tag(b)
			a = set(a)
			b = set(b)
			# print a,b
			try:
				inter_len = float(len(list(a.intersection(b))))
				union_len = float(len(list(a.union(b))))
				print "First String: ",string[i]
				print "Second String: ",string[j]
				print "jaccard_distance: ",inter_len/union_len
			except Exception,e:
				print e
				print "First String: ",string[i]
				print "Second String: ",string[j]
				print "jaccard_distance: ", 0



def cosine_sim(string,n):
	tfidf = TfidfVectorizer(tokenizer=tokernize_removestop,stop_words = 'english',lowercase=True)
	y = tfidf.fit_transform(string)
	y_array = (y * y.T).toarray()
	for i in xrange(0,n):
		for j in xrange(i+1,n):
			print "First String: ",string[i]
			print "Second String: ",string[j]
			print "cosine_sim: ", y_array[i][j] 


def main():
	n = int(raw_input('Enter no of string you will enter: '))
	input_array = []
	for i in xrange(0,n):
		input_array.append(raw_input())
		input_array[i] = input_array[i]
	try:
		print "jaccard_distance: "
		jaccard_distance(input_array,n)
	except Exception,e:
		print e
	try:
		print "cosine_similarity: "
		cosine_sim(input_array,n)
	except Exception,e:
		print e

main()