'''
Created on 21.01.2020

Doku NLTK:
https://www.nltk.org/api/nltk.corpus.html
http://www.nltk.org/nltk_data/
e.g.
nltk.corpus.gutenberg


word_tokenize():    zerlegen in word-list

@author: markus
'''
import os
import nltk
import nltk.corpus
import sys

# Datei entfernen
os.remove("Ausgabe.txt"); print("Datei entfernt")
# Ausgabe 1: 
print(os.listdir(nltk.data.find("corpora")), sep=' ')

# Ausgabe 1, Formatierung1
# for item in os.listdir(nltk.data.find("corpora")):
#    sys.stdout.write(item + ', ')

#===============================================================================
# # Ausgabe 2: Corpus Gutenberg
# print(nltk.corpus.gutenberg.fileids())  # 
# list1 = nltk.corpus.gutenberg.fileids()
# with open("Ausgabe.txt", "w") as afile:
#     afile.write("Corpus Gutenberg\n")
#     for item in list1:
#         afile.write(item+'\n')
#===============================================================================

# Ausgabe 2: Corpus Twitter
list1 = nltk.corpus.twitter_samples.fileids()
print("Twitter", list1)  # 
with open("Ausgabe.txt", "a") as afile:
    afile.write("\nCorpus Twitter Samples\n")
    for item in list1:
        afile.write(item+'\n')
        
# Ausgabe 2: Corpus general: Twitter
corpus = "twitter_samples"
list1 = list("nltk.corpus." + corpus + ".fileids()")
list1 = nltk.corpus.twitter_samples.fileids()
print(corpus, list1)  # 
with open("Ausgabe.txt", "a") as afile:
    afile.write("\nCorpus " + corpus + "\n")
    for item in list1:
        afile.write(item+'\n')
        
# Ausgabe 2: Corpus general; Stopwords
corpus = "stopwords"
list1 = "nltk.corpus." + corpus + ".fileids()"
list1 = nltk.corpus.stopwords.fileids()
print(corpus, list1)  # 
with open("Ausgabe.txt", "a") as afile:
    afile.write("\nCorpus " + corpus + "\n")
    for item in list1:
        afile.write(item+'\n')

# Ausgabe 3:    Hamlet
corpus = 'hamlet'
hamlet=nltk.corpus.gutenberg.words('shakespeare-hamlet.txt')
list1 = hamlet[:500]          # hamlet
print(hamlet)
with open("Ausgabe.txt", "a") as afile:
    afile.write("\nCorpus " + corpus + "\n")
    for word in list1:
        afile.write(word + ' ')
for word in list1[:500]:
    print(word, sep=' ', end = ' ')

# tokenize
print(list1)
print(type(list1))

AI = """According to the father of Artificial Intelligence, John McCarthy, it is 'The science and engineering of making intelligent machines, especially intelligent computer programs.'
Artificial Intelligence has made its breakthrough in almost all existing areas from science to business. If we look at the Google Trends Data for AI, for the past five years it has been growing continuously.
What is AI?
The theory and development of computer systems able to perform tasks normally requiring human intelligence, such as visual perception, speech recognition, decision-making, and translation between languages.
History of AI

A brief history of AI is given here.
Applications of AI

Currently Artificial Intelligence is used in almost all the domain areas of science and technology in the world. Few of them are given below.
Computer Science: Computer Vision, Games, Speech Recognition, Virtual Reality, etc
Biomedical Science: Image analysis, Disease diagnosis and prediction, Neuroscience.
Robotics
Finance, Marketing, and it goes on.
Recent news on AI
AlphaGo is a narrow AI computer program developed by Alphabet Inc.’s Google DeepMind in London to play the board game Go. AlphaGO has won against human champions on the game Go."""
print(AI)
from nltk.tokenize import word_tokenize
AI_tokens = word_tokenize(AI)
print(AI_tokens), print(type(AI_tokens), len(AI_tokens))

# Frequency Distinc
from nltk.probability import FreqDist
fdist = FreqDist()
for word in AI_tokens:
    fdist[word.lower()]+=1
print(fdist)
# item + count !missing!
print(FreqDist(AI_tokens))

fdist_top10 = fdist.most_common(10)
print(fdist_top10)

# number paragraphs: blankline_tokenize
from nltk.tokenize import blankline_tokenize
AI_blank = blankline_tokenize(AI)
print(len(AI_blank), AI_blank)      

# Bigrams, Trigrams, Ngrams
from nltk.util import bigrams, trigrams, ngrams
string = "The best and most beautiful things in the world cannot be seen or even touched, they must be felt with the heart"
quotes_tokens = nltk.word_tokenize(string)
print(quotes_tokens)
quotes_bigrams = list(nltk.bigrams(quotes_tokens))
quotes_trigrams = list(nltk.trigrams(quotes_tokens))
quotes_ngrams = list(nltk.ngrams(quotes_tokens, 4))
print(quotes_bigrams)
print(quotes_trigrams)
print(quotes_ngrams)

string = "Die besten und schönsten Dinge in der Welt können nicht gesehen oder berührt werden, sie müssen mit dem Herz gefühlt werden"
quotes_tokens = word_tokenize(string, 'german', False)
print(quotes_tokens)

# Stemming: PorterStemmer
from nltk.stem import PorterStemmer as PS
pst = PS()
print("Stemmer: ", pst.stem("having"))
        
words_to_stem = ["give", "giving", "given", "gave"]
for words in words_to_stem:
    print(words + ":" + pst.stem(words))
    
# Stemming: LancesterSTemmer (e.g.: wie oft wird give gebraucht?)
from nltk.stem import LancasterStemmer
lst = LancasterStemmer()
print("LancasterStemmer")
for words in words_to_stem:
    print(words+ ":" + lst.stem(words))
    
#===============================================================================
# # Lemmatization
# '''
# - groups together differend inflected forms of a word, called Lemma
# - similiar to stemming , as it maps into one common root
# '''
# from nltk.stem import wordnet
# from nltk.stem import WordNetLemmatizer
# word_lem = WordNetLemmatizer()
# print("Lemmatizer: WordNetLemmatizer")
# for words in words_to_stem:
#     print(words + ":" + word_lem.lemmatize(words))
#===============================================================================
    
# Stopwords
'''
nützlich für: 
nicht nützlich:        processing
'''

from nltk.corpus import stopwords, movie_reviews
print("Stopwords")
print(stopwords.words('english'))

import re
punctuation = re.compile(r'[-.?!,:;()|0-9]')
post_punctuation=[]
for words in AI_tokens:
    word = punctuation.sub("", words)
    if len(word)>0:
        post_punctuation.append(word)
print("post Punctuation:")
print(post_punctuation)
print(len(post_punctuation))

# POS: Parts of Speech
'''
grammar, verbs, questions, 

NN:    noun
VB:    Verb
JJ:    Adjective (e.g. natural)
DT:    Determiner (e.g. 'a', 'the') -> Artikel
TO:    to
'''

sent = "Timothy is a natural when it comes to drawing"
sent_tokens = word_tokenize(sent)
print("POS: Tags Tokens")
for token in sent_tokens:
    print(nltk.pos_tag([token]))
    
# NER: name entity recognition
'''
movie, monetary value, organization, location, quantities, person '''
    
print("NER: name entity recognition ")
from nltk import ne_chunk
NE_sent = "The US President stays in the WHITE HOUSE"
NE_tokens = word_tokenize(NE_sent)
NE_tags = nltk.pos_tag(NE_tokens)
NE_NER = ne_chunk(NE_tags)
print(NE_NER)

# Syntax
'''
   Phase Structure 
'''

new = "The big cat ate the little mouse who was after fresh cheese"
new_tokens = nltk.pos_tag(word_tokenize(new))
print("new tokens")
print(new_tokens)

grammar_np = r"NP: {<DT>?<JJ>*<NN>}"        # delimiter - ? - Adj - * - Noun
chunk_parser = nltk.RegexpParser(grammar_np)

chunk_result = chunk_parser.parse(new_tokens)
print("Chunk_result")
print(chunk_result)

# ------------------- MODEL ------------ 32:00 -------
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
print(os.listdir(nltk.data.find("corpora")))

from nltk.corpus import movie_reviews
print(movie_reviews.categories())

print(len(movie_reviews.fileids('pos')))
print(' ')
print(movie_reviews.fileids('pos'))

neg_rev = movie_reviews.fileids('neg')
print(neg_rev)
rev = nltk.corpus.movie_reviews.words('pos/cv000_29590.txt')
print(rev)

rev_list = []
for rev in neg_rev:
    rev_text_neg = rev = nltk.corpus.movie_reviews.words(rev)
    review_one_string = " ".join(rev_text_neg)
    review_one_string = review_one_string.replace(' ,', ',')
    review_one_string = review_one_string.replace(' .','.')
    review_one_string = review_one_string.replace("\' ", "'")
    review_one_string = review_one_string.replace(" \'", "'")
    rev_list.append(review_one_string)
print(len(rev_list))

neg_rev = movie_reviews.fileids('neg')
print(neg_rev)
rev = nltk.corpus.movie_reviews.words('pos/cv000_29590.txt')
print(rev)
print(len(rev_list))

pos_rev = movie_reviews.fileids('pos')
for rev_pos in pos_rev:
    rev_text_pos = nltk.corpus.movie_reviews.words(rev_pos)
    review_one_string = " ".join(rev_text_neg)
    review_one_string = review_one_string.replace(' ,', ',')
    review_one_string = review_one_string.replace(' .','.')
    review_one_string = review_one_string.replace("\' ", "'")
    review_one_string = review_one_string.replace(" \'", "'")
    rev_list.append(review_one_string)
print(len(rev_list))

neg_targets = np.zeros((1000,), dtype=np.int)
pos_targets = np.ones((1000,), dtype = np.int)
target_list = []
for neg_tar in neg_targets:
    target_list.append(neg_tar)
for pos_tar in pos_targets:
    target_list.append(pos_tar)
print(target_list)
y = pd.Series(target_list)
type(y)
print(y.head())

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(lowercase=True,stop_words='english',min_df=2)
X_count_vect = count_vect.fit_transform(rev_list)
print(X_count_vect.shape)
X_names = count_vect.get_feature_names()
print(X_names)