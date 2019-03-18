# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 16:48:24 2019

@author: Anuj
"""
#Import requied libraries

import pandas as pd
import numpy as np
import re
import os

from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras import optimizers

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer,WordNetLemmatizer
stemmer=SnowballStemmer('english')
lemma=WordNetLemmatizer()

import warnings
warnings.filterwarnings('ignore')

#Import dataset from folder location

path = "/Users/Anuj/Desktop/UIC courses/DS2/TakeHome - 1 - NLP/datasets/aclImdb/"
positiveFiles = [x for x in os.listdir(path+"train/pos/") if x.endswith(".txt")] #Training data with postive labels
negativeFiles = [x for x in os.listdir(path+"train/neg/") if x.endswith(".txt")] # Training data with negative labels
testFilespos = [x for x in os.listdir(path+"test/pos/") if x.endswith(".txt")] #Test data with postive labels
testFilesneg = [x for x in os.listdir(path+"test/neg/") if x.endswith(".txt")] #Test data with negative labels
positiveReviews, negativeReviews, testReviewspos, testReviewsneg = [], [], [], []
for pfile in positiveFiles:
    with open(path+"train/pos/"+pfile, encoding="latin1") as f:
        positiveReviews.append(f.read())
for nfile in negativeFiles:
    with open(path+"train/neg/"+nfile, encoding="latin1") as f:
        negativeReviews.append(f.read())
for t1file in testFilespos:
    with open(path+"test/pos/"+t1file, encoding="latin1") as f:
        testReviewspos.append(f.read())
for t2file in testFilesneg:
    with open(path+"test/neg/"+t2file, encoding="latin1") as f:
        testReviewsneg.append(f.read())
        
reviews = pd.concat([
    pd.DataFrame({"review":positiveReviews, "label":1, "file":positiveFiles}), #Training dataset with positive and negative labels
    pd.DataFrame({"review":negativeReviews, "label":0, "file":negativeFiles})
], ignore_index=True).sample(frac=1, random_state=1)

test_reviews = pd.concat([pd.DataFrame({"review":testReviewspos, "label":1, "file":testFilespos}), # Test dataset with positive and negative labels
    pd.DataFrame({"review":testReviewsneg, "label":0, "file":testFilesneg})
], ignore_index = True).sample(frac = 1, random_state = 1)

reviews = reviews[["review", "label", "file"]].sample(frac=1, random_state=1)
test_reviews = test_reviews[["review", "label", "file"]].sample(frac=1, random_state=1)
train = reviews[reviews.label!=-1].sample(frac=0.6, random_state=1)
valid = reviews[reviews.label!=-1].drop(train.index)
test = test_reviews[test_reviews.label != -1]

print('\nTraining data shape:', train.shape)
print('\nValidation data shape:', valid.shape)
print('\nTesting data shape:', test.shape)

stopwords = nltk.corpus.stopwords.words('english')

#Function to clean text includes lowercase, lemmatization and removing stopwords
 
def clean_review(review_col):
    review_corpus=[]
    for i in range(0,len(review_col)):
        review=str(review_col[i])
        review=re.sub('[^a-zA-Z]',' ',review)
        #review=[stemmer.stem(w) for w in word_tokenize(str(review).lower())]
        review=[lemma.lemmatize(w) for w in word_tokenize(str(review).lower())]
        review_nosw = [word for word in review if word not in stopwords]
        review=' '.join(review_nosw)
        review_corpus.append(review)
    return review_corpus

train = train.reset_index(drop = True)
valid = valid.reset_index(drop = True)

#Cleaning train and validation dataset
train['clean_review'] = clean_review(train.review)
valid['clean_review'] = clean_review(valid.review)

trainX = train.clean_review
validX = valid.clean_review

#Tfid vectorizer to vectorize the text, retain 10000 features per text and with document frequency cut-off 2
#10000 features found to be sufficient to get good accuracy

tfidf = TfidfVectorizer(min_df=2, max_features=10000, stop_words=stopwords) #, ngram_range=(1,3)

#Fit and transform tfidf on trainX but only transform tfidf on validX

trainX = tfidf.fit_transform(trainX).toarray()
validX = tfidf.transform(validX).toarray()

#Target values

trainY = train.label
validY = valid.label

#Feature selection
#Keep the 2000 features out of 10k that correlate the most with the target

from scipy.stats.stats import pearsonr
getCorrelation = np.vectorize(lambda x: pearsonr(trainX[:,x], trainY)[0])
correlations = getCorrelation(np.arange(trainX.shape[1]))

allIndeces = np.argsort(-correlations)
bestIndeces = allIndeces[np.concatenate([np.arange(1000), np.arange(-1000, 0)])]

vocabulary = np.array(tfidf.get_feature_names())

trainX = trainX[:,bestIndeces]
validX = validX[:,bestIndeces]

#MODEL 1 - NEURAL NETWORK
#Simple neural network model with 6 layers + input layer

DROPOUT = 0.5 #Fraction of input units to be dropped during each layer - prevents overfitting
ACTIVATION = "tanh"

model = Sequential([    
    Dense(int(trainX.shape[1]/2), activation=ACTIVATION, input_dim=trainX.shape[1]),
    Dropout(DROPOUT),
    Dense(int(trainX.shape[1]/2), activation=ACTIVATION, input_dim=trainX.shape[1]),
    Dropout(DROPOUT),
    Dense(int(trainX.shape[1]/4), activation=ACTIVATION),
    Dropout(DROPOUT),
    Dense(100, activation=ACTIVATION),
    Dropout(DROPOUT),
    Dense(20, activation=ACTIVATION),
    Dropout(DROPOUT),
    Dense(5, activation=ACTIVATION),
    Dropout(DROPOUT),
    Dense(1, activation='sigmoid'),
])
    
model.compile(optimizer=optimizers.Adam(0.00005), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

EPOCHS = 30
BATCHSIZE = 1500

#Fitting neural network model on training data and checking accuracy on train and validation data
print('Fitting neural network\n')
model.fit(trainX, trainY, epochs=EPOCHS, batch_size=BATCHSIZE, validation_data=(validX, validY))

#Plotting model accuracy and model loss

history = model.history
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#To check accuracy
#Evaluating model on training dataset
train["probability"] = model.predict(trainX)
train["prediction"] = train.probability-0.5>0
train["truth"] = train.label==1

print('\nAccuracy of Neural Network on training set:', (train.truth==train.prediction).mean())

#Evaluating model on validation dataset
valid["probability"] = model.predict(validX)
valid["prediction"] = valid.probability-0.5>0
valid["truth"] = valid.label==1

#MODEL 2 - LINEAR SVC
#MODEL 3 - MULTINOMIAL NAIVE BAYES 
#MODEL 4 - RANDOM FOREST CLASSIFIER

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

#Fit the model to training dataset
svc = LinearSVC().fit(trainX, trainY)
clf = MultinomialNB().fit(trainX, trainY)
rfc = RandomForestClassifier(n_jobs = -1, n_estimators = 100).fit(trainX, trainY)

#Predicting training labels 
pred_train_svc = svc.predict(trainX)
pred_train_clf = clf.predict(trainX)
pred_train_rfc = rfc.predict(trainX)

from sklearn.metrics import accuracy_score

#Checking accuracy on training dataset
print('Accuracy of LinearSVC on training set:', accuracy_score(pred_train_svc, trainY))
print('Accuracy of MultinomialNB on training set:', accuracy_score(pred_train_clf, trainY))
print('Accuracy of Random Forest Classifier on training set:', accuracy_score(pred_train_rfc, trainY))

#Predicting validation labels
pred_val_svc = svc.predict(validX)
pred_val_clf = clf.predict(validX)
pred_val_rfc = rfc.predict(validX)

#Checking accuracy on validation dataset
print('\nAccuracy of Neural Network on validation set:', (valid.truth==valid.prediction).mean())
print('Accuracy of LinearSVC on validation set:', accuracy_score(pred_val_svc, validY))
print('Accuracy of MultinomialNB on validation set:', accuracy_score(pred_val_clf, validY))
print('Accuracy of RandomForestClassifier on validation set:', accuracy_score(pred_val_rfc, validY))

#Preprocessing test dataset
test = test.reset_index(drop = True)

#Cleaning test dataset
test['clean_review'] = clean_review(test.review)

testX = test.clean_review
testX = tfidf.transform(testX).toarray()
testY = test.label
testX = testX[:,bestIndeces]
           
#Predicting test labels using the three models
probability = model.predict(testX)[0,0] #Neural Network
unseen_pred_svc = svc.predict(testX) #LinearSVC
unseen_pred_clf = clf.predict(testX) #MultinomalNB
unseen_pred_rfc = rfc.predict(testX) #Random Forest Classification

#Evaluating neural network model on test dataset
test["probability"] = model.predict(testX)
test["prediction"] = test.probability-0.5>0
test["truth"] = test.label==1

#Checking accuracy on test dataset
print('\nAccuracy of Neural Network on test set:', (test.truth==test.prediction).mean()) #Neural network
print('Accuracy of LinearSVC on test set:', accuracy_score(unseen_pred_svc, testY)) #LinearSVC
print('Accuracy of MultinomialNB on test set:', accuracy_score(unseen_pred_clf, testY)) #MultinomialNB
print('Accuracy of RandomForestClassifier on test set:', accuracy_score(unseen_pred_rfc, testY)) #Random Forest Classifier