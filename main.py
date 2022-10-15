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


<<<<<<< Updated upstream
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
=======
# def affichageResultat():
#     print('Test Result:\n')
#     print('===================================================')
#     print('Accuracy Score: ' + str(accuracy_score(X_train)) + ' %')
#     print('___________________________________________________')
>>>>>>> Stashed changes


random_state = 42

<<<<<<< Updated upstream
data = pd.read_csv('Titanic_Research_v6.csv', sep=';')
print(data.shape)
sns.countplot(x=data["survived"])
plt.show()
data = data.drop(columns=['body'])
print(data.shape)

print(data.dtypes)

categorical_col = data.select_dtypes('object')

for i in categorical_col.columns:
    le = preprocessing.LabelEncoder()
    le.fit_transform(categorical_col[i])
    data[i] = le.transform(data[i])
print(data.shape)

X = data.drop(columns='survived')
y = data["survived"]
plt.show()
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=random_state)

print(X_train.shape)
print(X_test.shape)
print(data.isnull().sum())

max_depth_range = [5]
param_grid = {'criterion': ["gini"], 'splitter': ["best"], 'max_depth': max_depth_range}
clf = GridSearchCV(DecisionTreeClassifier(random_state=random_state), param_grid, scoring="accuracy", n_jobs=-1,
                   verbose=1, cv=3)
clf.fit(X_train, y_train)
best_params_list = clf.best_params_
print(best_params_list)
classifier = DecisionTreeClassifier(criterion=best_params_list['criterion'], max_depth=best_params_list['max_depth'],
                                    splitter=best_params_list['splitter'], random_state=random_state)
classifier.fit(X_train, y_train)
y_pred_train = classifier.predict(X_train)
affichage_resultat(y_train, y_pred_train, 'Train')
y_pred_test = classifier.predict(X_test)
affichage_resultat(y_test, y_pred_test, 'Test')
plot_tree(classifier, feature_names=X.columns, filled=True, rounded=True)
plt.savefig("Tree.svg")
plt.show()

param_grid = {'criterion': ["gini", "entropy"], 'max_depth': max_depth_range}
clf = GridSearchCV(RandomForestClassifier(random_state=random_state), param_grid, scoring="accuracy", n_jobs=-1,
                   verbose=1, cv=3)
clf.fit(X_train, y_train)
best_params_list = clf.best_params_
print(best_params_list)
classifier = RandomForestClassifier(criterion=best_params_list['criterion'], max_depth=best_params_list['max_depth'],
                                    random_state=random_state)
classifier.fit(X_train, y_train)
y_pred_train = classifier.predict(X_train)
affichage_resultat(y_train, y_pred_train, 'Train')
y_pred_test = classifier.predict(X_test)
affichage_resultat(y_test, y_pred_test, 'Test')

# plot_tree(classifier)
# plt.savefig("Forest.svg")
# plt.show()
=======
data = pd.read_csv('Titanic_Research_v6.csv')
# print(data.shape)
# sns.countplot(x=data)
plt.show()
# data = data.drop(columns=['Over18', 'StandardHours', 'EmployeeNumber', 'EmployeeCount'])
# print(data.shape)

print(data.dtypes)

# categorical_col = data.select_dtypes('object')
#
# for i in categorical_col.columns:
#     le = preprocessing.LabelEncoder()
#     le.fit_transform(categorical_col[i])
#     data[i] = le.transform(data[i])
# print(data.shape)
#
# X = data.drop(columns='Attrition')
# y = data["Attrition"]
# print(X.shape)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=random_state)
#
# print(X_train.shape)
# print(X_test.shape)
#
# classifier = DecisionTreeClassifier(random_state=random_state).fit(X_train, y_train)
#
# # accuracy_score(X_train)
>>>>>>> Stashed changes
