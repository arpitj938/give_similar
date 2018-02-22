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
from nltk.corpus import wordnet

reload(sys)  
sys.setdefaultencoding('utf8')

lemmatizer = WordNetLemmatizer()
stops = set(stopwords.words("english"))      


# POS tag uses treebank_tag eg: noun plural is NNP noun singular is NNS but lemmatizer don't accept those
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''        

# Tokenize (task done are : lower, remove puncutation , tokenize , added tag , lemmatize , remove stop word)
def tokernize_removestop(a):
	a = a.lower()
	# print a
	a = re.sub('[():?.,]',"",a)
	# print a
	a = word_tokenize(a)
	# print "token: ", a
	b = add_pos_tag(a)[0]
	# print "pos_tag_sents: ",b
	c = []
	for w in b:
		first = w[0]
		second = w[1]
		# print first,second
		try:
			c.append(lemmatizer.lemmatize(first,get_wordnet_pos(second)))
		except Exception,e:
			c.append(first)
	a =  c
	# print "lemmatize: ", a
	c = a
	a = [w for w in a if not w in stops]
	if(len(a)==0):
		a=c
	# print "token",a
	return a


#Added Tag in sentances like noun, verb etc..		
def add_pos_tag(a):
	return nltk.pos_tag_sents([a])

#jaccard Distance
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
			print a,b
			try:
				inter_len = float(len(list(a.intersection(b))))
				union_len = float(len(list(a.union(b))))
				print "First String: ",string[i]
				print "Second String: ",string[j]
				jaccard_distance_ans = inter_len/union_len
				print "jaccard_distance: ",jaccard_distance_ans
			except Exception,e:
				print e
				print "First String: ",string[i]
				print "Second String: ",string[j]
				jaccard_distance_ans =0
				print "jaccard_distance: ",jaccard_distance_ans
			if(jaccard_distance_ans>0.9):
				print "Similar"
			else:
				print "Disimilar"


# Cosine Distance 
def cosine_sim(string,n):
	tfidf = TfidfVectorizer(tokenizer=tokernize_removestop,stop_words = 'english',lowercase=True)
	y = tfidf.fit_transform(string)
	y_array = (y * y.T).toarray()
	for i in xrange(0,n):
		for j in xrange(i+1,n):
			print "First String: ",string[i]
			print "Second String: ",string[j]
			print "cosine_sim: ", y_array[i][j] 
			if(y_array[i][j]>0.9):
				print "Similar"
			else:
				print "Disimilar"

#Main function
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