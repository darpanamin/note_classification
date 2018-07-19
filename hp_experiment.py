# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 13:06:33 2018

@author: DCA
"""

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

from IPython.display import Image  
from sklearn.metrics import classification_report
import pickle
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn import svm
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt


train_vec = pd.read_csv('C:\\Users\\DCA\\Dropbox\\python\\datasets\\datasets\\count_vector.csv',encoding = "ISO-8859-1",sep = "|")
train_vec_2018 = pd.read_csv('C:\\Users\\DCA\\Dropbox\\python\\datasets\\datasets\\count_vector_2018.csv',encoding = "ISO-8859-1",sep = ",")


train_vec.head(5)
train_vec = train_vec.iloc[:,0:27]

#drop null target rows
train_vec_fin = train_vec.dropna()

train_features = train_vec_fin.iloc[:,2:27]
train_features_2018 = train_vec_2018.iloc[:,2:27]


train_feat_matrix = train_features.as_matrix()
train_feat_matrix_2018 = train_features_2018.as_matrix()

X = train_feat_matrix 
X_2018 = train_feat_matrix_2018
X_COMB = np.concatenate((X, X_2018), axis=0)   


Y = train_vec_fin['TARGET']
Y_2018 = train_vec_2018['TARGET']
Y_COMB = Y.append(Y_2018)


clf = tree.DecisionTreeClassifier()

clf = clf.fit(X, Y)
clf_2018 = clf.fit(X_2018, Y_2018)
clf_comb = clf.fit(X_COMB, Y_COMB)

pred_results = clf.predict(X)
pred_results = clf_comb.predict(X)
pred_results = clf_2018.predict(X)

metrics.accuracy_score(Y, pred_results)
print(classification_report(Y, pred_results))
pred_prob = clf_comb.predict_proba(X)
print(pred_prob[:,1])
min(pred_prob[:0,1])


clf_save = clf
clf_save__tree2 = clf_comb
clf_save_mlp3 = clf_comb
clf_save_mlp4 = clf_comb # adam
clf_save_mlp5 = clf_comb # lbfgs

pred_prob = clf.predict_proba(X[3])

mlp = 0
mlp = MLPClassifier(hidden_layer_sizes=(28),solver='sgd',learning_rate_init=0.01,max_iter=700)
mlp = MLPClassifier(hidden_layer_sizes=(28),solver='adam',learning_rate_init=0.01,max_iter=700)
mlp = MLPClassifier(hidden_layer_sizes=(28),solver='lbfgs',learning_rate_init=0.01,max_iter=700)

clf_comb = 0
clf = mlp.fit(X,Y)
clf_2018 = mlp.fit(X_2018,Y_2018)
clf_comb = mlp.fit(X_COMB,Y_COMB)

joblib.dump(clf_comb, 'MLP_adam_2018_june.pkl') 
joblib.dump(clf_comb, 'MLP_lbfgs_2018_june.pkl') 

clf_arch = joblib.load('NNMLP_model.pkl') 

clf.classes_
##### 2018 Testing
pred_results_2018 = clf.predict(X_2018)
pred_results_2018 = clf_comb.predict(X_2018)
metrics.accuracy_score(Y_2018, pred_results_2018)
print(classification_report(Y_2018, pred_results_2018))

##### combined testing

pred_results_comb = clf_comb.predict(X_COMB)
metrics.accuracy_score(Y_COMB, pred_results_comb)
print(classification_report(Y_COMB, pred_results_comb))

##### new approach ##################

train_targets = pd.read_csv('C:\\Users\\DCA\\Dropbox\\python\\datasets\\datasets\\Jun_Jul_Targets.csv',encoding = "ISO-8859-1",sep = ",")
train_text = pd.read_csv('C:\\Users\\DCA\\Dropbox\\python\\datasets\\datasets\\Jun_Jul_Text.csv',encoding = "ISO-8859-1",sep = ",")
train_text_2 = train_text

train_targets.iloc[:,1]

train_text_2 = train_text_2.groupby(['NOTE_ID'])['NOTE_TEXT'].apply(' '.join).reset_index()

train_text_2['NOTE_TEXT'].str.len()
train_text_2 = pd.merge(train_text_2, train_targets, on='NOTE_ID')

ng_range=(1, 3)

vectorizer = TfidfVectorizer( ngram_range=ng_range,sublinear_tf=True, max_df=.98,min_df=.1,stop_words='english')
X_train = vectorizer.fit_transform(train_text_2['NOTE_TEXT'])

print(len(vectorizer.vocabulary_))
feature_names = vectorizer.get_feature_names()
df_features = pd.DataFrame(feature_names)
df_features[df_features[0].str.contains("pro")] 


#clf=SGDClassifier(penalty='elasticnet')
clf=LogisticRegression(penalty='l2')
list(train_text_2.columns.values)
'chief_complaint',
 'HPI',
 'MED_HIST',
 'SURG_HIST',
 'Medications',
 'Allergies',
 'SOCIAL_HISTORY',
 'FAMILY_HISTORY',
 'ROS',
 'PHYS_EXAM_CARDIO',
 'PHYS_EXAM_PULM'

Y_train_CCP = train_text_2["chief_complaint"]
Y_train_HPI = train_text_2["HPI"]
Y_train_MHX = train_text_2["MED_HIST"]
Y_train_SRG = train_text_2["SURG_HIST"]
Y_train_MED = train_text_2["Medications"]
Y_train_ALG = train_text_2["Allergies"]
Y_train_SOC = train_text_2["SOCIAL_HISTORY"]
Y_train_FAM = train_text_2["FAMILY_HISTORY"]
Y_train_ROS = train_text_2["ROS"]
Y_train_PEC = train_text_2["PHYS_EXAM_CARDIO"]
Y_train_PEP = train_text_2["PHYS_EXAM_PULM"]

Y_train = Y_train_CCP
scores_prec = cross_val_score(clf, X_train, Y_train, cv=3, scoring='precision')
scores_acc = cross_val_score(clf, X_train, Y_train, cv=3, scoring='accuracy')
scores_f1 = cross_val_score(clf, X_train, Y_train, cv=3, scoring='f1')
print(np.mean(scores_prec),"\n",np.mean(scores_acc),np.mean(scores_f1))

clf_CCP = clf.fit(X_train,Y_train_CCP)
clf_HPI = clf.fit(X_train,Y_train_HPI)
clf_MHX = clf.fit(X_train,Y_train_MHX)

predicted = cross_val_predict(clf, X_train, Y_train, cv=10,method="predict_proba") 
predicted = cross_val_predict(clf, X_train, Y_train, cv=10)
 
scores = cross_val_score(mlp, X_train, Y_train, cv=3, scoring='precision')

preds = cross_val_predict(clf, X_train, Y_train, cv=9)

print(np.min(scores))

print(clf.coef_)
feat_coef = clf.coef_

clf_ros = clf.fit(X_train,Y_train)
clf_pec = clf.fit(X_train,Y_train)
pred_sgd_ros = clf_ros.predict(X_train)
metrics.accuracy_score(Y_train, pred_sgd_ros)
print(classification_report(Y_train, preds))


clf_temp = clf_HPI
coef = clf_temp.coef_.ravel()

coef_feat_pd = pd.DataFrame({'coef': coef,'feature_name': feature_names})
coef_feat_pd[coef_feat_pd['feature_name'].str.contains("chief", regex=True)].head(55)
coef_feat_pd[coef_feat_pd['coef'] >.2].head(55)

coef_feat_pd[coef_feat_pd['coef'] <-.2].head(55)

