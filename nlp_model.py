'''
Created on 21.01.2020

Doku NLTK:
https://www.nltk.org/api/nltk.corpus.html
http://www.nltk.org/nltk_data/
e.g.
nltk.corpus.gutenberg


word_tokenize():    zerlegen in word-list

libraries:
- nltk
- pandas
- sklearn

@author: markus
'''
import os
import nltk
import nltk.corpus
import sys
import time


# ------------------- MODEL ------------ 32:00 -------
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
print(os.listdir(nltk.data.find("corpora"))) #------- ['stopwords', 'sentence_polarity', ..]

from nltk.corpus import movie_reviews
print(movie_reviews.categories()) # ---['neg', 'pos']

# fileIDs pos
print(' fileIds pos')
print("len : ", len(movie_reviews.fileids('pos'))) # 1000
print(movie_reviews.fileids('pos'))
eingabe = input('Warten')

# fileIDs neg/pos    / words(pos)
neg_rev = movie_reviews.fileids('neg')          
print('neg_rev: ', neg_rev)
rev = movie_reviews.fileids('pos')
print('rev: ', rev)
rev = nltk.corpus.movie_reviews.words('pos/cv000_29590.txt')  # ?
print("rev: ", rev, "len(rev): ", len(rev))
eingabe = input('Warten')

# corpora_words
rev_list = []
# for testing
for rev in neg_rev:
    review_one_stringt = " ".join(rev)
with open ("Ausgabe1.txt", "w") as ausgabe1:
    ausgabe1.write("Review One String\n" + str(review_one_stringt))
    ausgabe1.write(str(rev_list)+"\n\n\n")


for rev in neg_rev:
    rev_text_neg = rev = movie_reviews.words(rev)
    review_one_string = " ".join(rev_text_neg)
    review_one_string = review_one_string.replace(' ,', ',')
    review_one_string = review_one_string.replace(' .','.')
#    review_one_string = review_one_string.replace("\' ", "'")
#    review_one_string = review_one_string.replace(" \'", "'")
    rev_list.append(review_one_string)
# print("rev_list", rev_list)
with open ("Ausgabe1.txt", "a") as ausgabe1:
    ausgabe1.write("Review One String2\n")
    ausgabe1.write(str(rev_list))
    
print("len rev_list :", len(rev_list))
eingabe = input('Warten1')


neg_rev = movie_reviews.fileids('neg')
print(neg_rev)
rev = nltk.corpus.movie_reviews.words('pos/cv000_29590.txt')
print("rev: ", rev)
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
print("len rev_list: ", len(rev_list))
eingabe = input("warten2")

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

# eingabe = input("Weitermachen ?")
print(X_names)
X_count_vect = pd.DataFrame(X_count_vect.toarray(), columns=X_names)
X_count_vect.shape   # (2000, 16228)
print(X_count_vect.head())

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
X_train_cv,X_test_cv,y_train_cv,y_test_cv = train_test_split(X_count_vect,y, test_size=0.25,random_state=5)
print(X_train_cv.shape)
print(X_test_cv.shape)

from sklearn.naive_bayes import GaussianNB   # learn classifier
gnb = GaussianNB()
y_pred_gnb = gnb.fit(X_train_cv, y_train_cv).predict(X_test_cv)
from sklearn.naive_bayes import MultinomialNB
clf_cv = MultinomialNB()
print(clf_cv.fit(X_train_cv,y_train_cv))

y_pred_cv = clf_cv.predict(X_test_cv)
print(type(y_pred_cv))

print(metrics.accuracy_score(y_test_cv,y_pred_cv))   # accuracy: 1.0: overfitting
score_clf_cv = confusion_matrix(y_test_cv, y_pred_cv)
print(score_clf_cv)

