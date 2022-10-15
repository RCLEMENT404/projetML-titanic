# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 20:25:04 2022

@author: ahmed
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing 
#################################################################
#################################################################
##Afficher et importer data
#################################################################
#################################################################


data=pd.read_csv("Tested.csv")
data2=pd.read_csv("Titanic.csv")
data3=pd.read_csv("Titanic-Dataset.csv")
data45=pd.read_csv("Titanic_Research_v6.csv")
print(data)
print(data2)
print(data3)
print(data4)

print(data['Cabin'])

data4 = pd.DataFrame(data,columns=['Cabin'])

data5=data4.select_dtypes('object')

randome_state=42

for i in data5:
    le = preprocessing.LabelEncoder()
    le.fit_transform(data5[i])
    data5[i]=le.transform(data5[i])
print(data.shape)
    

data.shape
dim = data.shape
print(dim[0]) # 228
print(dim[1]) # 4

data.dtypes

print(data.dtypes)

###################################################################
###################################################################
## choisir attrition et le plot
###################################################################
###################################################################


# data2=data['Attrition']

# print(data2)

# sns.countplot(data2)
# plt.show
# plt.savefig('image1.png')

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import LabelEncoder
# #################################################################
# #################################################################
# ##Afficher et importer data
# #################################################################
# #################################################################


# data=pd.read_csv("HR-Employee-Attrition.csv") 

# print(data)

# data.shape
# dim = data.shape
# print(dim[0]) # 228
# print(dim[1]) # 4

# data.dtypes

# print(data.dtypes)

# ###################################################################
# ###################################################################
# ## choisir attrition et le plot
# ###################################################################
# ###################################################################


# a=data['Attrition']
# n=data["Age"]
# print(a)
# print(n)


# plt.figure(1,figsize=(12,8))
# sns.countplot(a, label='oui')
# plt.legend()
# plt.show
# plt.savefig('image1.png')


# # plt.figure(2,figsize=(12,8))
# # sns.countplot(n, color='red', log=True)

# # plt.show

# ## essai
# print("dataframe")
# data2 = pd.DataFrame(data,columns=['Attrition','Age'])
# print(data2)
# # df_data2=data2.values


# # print("`\n\n dataframe to array")
# # print(df_data2)

# data2=data.drop(['Age','Attrition'], axis=1) # suprimer colones



# acategorical_col=pd.DataFrame(data,columns=['Attrition','BusinessTravel','Department ','EducationField ','Gender',
#                  'JobRole ','MaritalStatus','OverTime'])


# def Encoder(categorical_col):
#           columnsToEncode = list(categorical_col.select_dtypes(include=['category','object']))
#           le = LabelEncoder()
#           for feature in columnsToEncode:
#               try:
#                   categorical_col[feature] = le.fit_transform(categorical_col[feature])
#               except:
#                   print('Error encoding '+feature)
#           return categorical_col

# Encoder(categorical_col)
# print(Encoder(categorical_col))
# categorical_col.dtypes

# print(categorical_col.dtypes)


# data3=data
# data3=data[categorical_col]

# # label_encoder=LabelEncoder()

# # data3=label_encoder.fit_transform(data['Attrition','BusinessTravel','Department ','EducationField ','Gender',
# #                  'JobRole ','MaritalStatus','Over18','OverTime'])