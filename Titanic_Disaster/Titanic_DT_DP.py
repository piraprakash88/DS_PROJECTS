# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 12:40:38 2019

@author: Pira
"""
import os
import pydot #if we need to use any external .exe files.... Here we are using dot.exe

import io #For i/o operations
import pandas as pd
from sklearn import tree #For Decissin Tree
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

#Read Train Data file
titanic_train = pd.read_csv("E:\\DATASCIENCE\\Titanic_Project\\train.csv")

titanic_train.shape #Not mandatory though!!
titanic_train.info()

titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
mean_imputer = preprocessing.Imputer()
mean_imputer.fit(titanic_train1[['Age','Fare']])
titanic_train1[['Age','Fare']] = mean_imputer.transform(titanic_train1[['Age','Fare']])
titanic_train1.shape
titanic_train1.info()
titanic_train1.describe
X_train = titanic_train1.drop(['PassengerId','Cabin','Ticket', 'Name','Survived'],1) 
y_train = titanic_train['Survived']
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
dt = tree.DecisionTreeClassifier()
param_grid = {'max_depth':[8, 10, 15], 'min_samples_split':[2, 4, 6], 'criterion':['gini', 'entropy']}

print(type(param_grid))
dt_grid = model_selection.GridSearchCV(dt, param_grid, cv=6, n_jobs=8)

dt_grid.fit(X_train,y_train)

type(dt)
#dt_grid.grid_scores_
#dt_grid.cv_results_
dt_grid.best_params_
dt_grid.best_score_ 

titanic_test = pd.read_csv("E:\\DATASCIENCE\\Titanic_Project\\test.csv")
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()
titanic_test.Fare[titanic_test['Age'].isnull()] = titanic_test['Age'].mean()
titanic_test1 = pd.get_dummies(titanic_test, columns=['Pclass', 'Sex', 'Embarked'])
X_test = titanic_test1.drop(['PassengerId','Age','Cabin','Ticket', 'Name'], 1)
X_test = sc.fit_transform(X_test)
titanic_test1.info()
titanic_test['Survived'] = dt_grid.predict(X_test)
titanic_test.to_csv("Submission_Grid.csv", columns=['PassengerId', 'Survived'], index=False)