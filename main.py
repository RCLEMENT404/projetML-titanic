# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 15:57:15 2022

@author: cleme
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def affichage_resultat(y_true, y_pred, result_type):
    print(str(result_type) + ' Result:\n')
    print('===================================================')
    print('Accuracy Score: ' + str(round(accuracy_score(y_true, y_pred), 4) * 100) + '%')
    print('____________________________________________________')
    print('CLASSIFICATION REPORT:')
    print(classification_report(y_true, y_pred))
    print('___________________________________________________')
    print('Confusion Matrix:')
    print(confusion_matrix(y_pred, y_true))


random_state = 42

data = pd.read_csv('HR-Employee-Attrition.csv')
print(data.shape)
sns.countplot(x=data["Attrition"])
plt.show()
data = data.drop(columns=['Over18', 'StandardHours', 'EmployeeNumber', 'EmployeeCount'])
print(data.shape)

print(data.dtypes)

categorical_col = data.select_dtypes('object')

for i in categorical_col.columns:
    le = preprocessing.LabelEncoder()
    le.fit_transform(categorical_col[i])
    data[i] = le.transform(data[i])
print(data.shape)

X = data.drop(columns='Attrition')
y = data["Attrition"]
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=random_state)

print(X_train.shape)
print(X_test.shape)

classifier = DecisionTreeClassifier(random_state=random_state)
classifier.fit(X_train, y_train)
y_pred_train = classifier.predict(X_train)
affichage_resultat(y_train, y_pred_train, 'Train')
y_pred_test = classifier.predict(X_test)
affichage_resultat(y_test, y_pred_test, 'Test')

max_depth_range = range(1, 100)
max_samples_range = range(2, 10)
param_grid = {'criterion': ["gini", "entropy"], 'splitter': ["best", "random"], 'max_depth': max_depth_range,
              'min_samples_split': max_samples_range, 'min_samples_leaf': max_samples_range,
              'random_state': [random_state]}
clf = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring="accuracy", n_jobs=-1, verbose=1, cv=3)
clf.fit(X_train, y_train)
best_params_list = clf.best_params_
print(best_params_list)
classifier = DecisionTreeClassifier(criterion=best_params_list['criterion'], max_depth=best_params_list['max_depth'],
                                    min_samples_split=best_params_list['min_samples_split'],
                                    min_samples_leaf=best_params_list['min_samples_leaf'],
                                    splitter=best_params_list['splitter'], random_state=random_state)
classifier.fit(X_train, y_train)
y_pred_train = classifier.predict(X_train)
affichage_resultat(y_train, y_pred_train, 'Train')
y_pred_test = classifier.predict(X_test)
affichage_resultat(y_test, y_pred_test, 'Test')
plot_tree(classifier)
plt.savefig("Tree.png")
plt.show()

classifier = RandomForestClassifier(random_state=random_state, n_estimators=100)
classifier.fit(X_train, y_train)
y_pred_train = classifier.predict(X_train)
affichage_resultat(y_train, y_pred_train, 'Train')
y_pred_test = classifier.predict(X_test)
affichage_resultat(y_test, y_pred_test, 'Test')

param_grid = {'criterion': ["gini", "entropy"], 'max_depth': max_depth_range,
              'min_samples_split': max_samples_range, 'min_samples_leaf': max_samples_range,
              'random_state': [random_state]}
clf = GridSearchCV(RandomForestClassifier(), param_grid, scoring="accuracy", n_jobs=-1, verbose=1, cv=3)
clf.fit(X_train, y_train)
best_params_list = clf.best_params_
print(best_params_list)
classifier = RandomForestClassifier(criterion=best_params_list['criterion'], max_depth=best_params_list['max_depth'],
                                    min_samples_split=best_params_list['min_samples_split'],
                                    min_samples_leaf=best_params_list['min_samples_leaf'], random_state=random_state)
classifier.fit(X_train, y_train)
y_pred_train = classifier.predict(X_train)
affichage_resultat(y_train, y_pred_train, 'Train')
y_pred_test = classifier.predict(X_test)
affichage_resultat(y_test, y_pred_test, 'Test')