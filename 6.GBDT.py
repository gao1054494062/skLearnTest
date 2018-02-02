import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics

import matplotlib.pylab as plt

train = pd.read_csv('E:\Python\\test\skLearn\\train_modified\\train_modified.csv')
target='Disbursed' # Disbursed的值就是二元分类的输出
IDcol = 'ID'
train['Disbursed'].value_counts() 

x_columns = [x for x in train.columns if x not in [target, IDcol]]
X = train[x_columns]
y = train['Disbursed']

#rf0 = RandomForestClassifier(oob_score=True, random_state=10)
#rf0.fit(X,y)
#print(rf0.oob_score_)
#y_predprob = rf0.predict_proba(X)[:,1]
#print("AUC Score (Train): ",metrics.roc_auc_score(y, y_predprob))

param_test1 = {'n_estimators':range(10,71,10)}
print(param_test1)
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,
                                  min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10), 
                       param_grid = param_test1, scoring='roc_auc',cv=5)
gsearch1.fit(X,y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_



param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(50,201,20)}
gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60, 
                                  min_samples_leaf=20,max_features='sqrt' ,oob_score=True, random_state=10),
   param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
gsearch2.fit(X,y)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

rf1 = RandomForestClassifier(n_estimators= 60, max_depth=13, min_samples_split=110,
                                  min_samples_leaf=20,max_features='sqrt' ,oob_score=True, random_state=10)
rf1.fit(X,y)
print(rf1.oob_score_)

y_predprob = rf0.predict_proba(X)[:,1]
print("AUC Score (Train): ",metrics.roc_auc_score(y, y_predprob))