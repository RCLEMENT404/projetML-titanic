import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def affichageResultat():
    print('Test Result:\n')
    print('===================================================')
    print('Accuracy Score: ' + str(accuracy_score(X_train)) + ' %')
    print('___________________________________________________')


random_state = 42

data = pd.read_csv('water_potability.csv')
print(data.shape)
sns.countplot(x=data)
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

classifier = DecisionTreeClassifier(random_state=random_state).fit(X_train, y_train)

# accuracy_score(X_train)
