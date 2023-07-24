import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("titanic_train.csv")
# print(df)
# print(len(df))
# print(df.head())

sns.countplot(x='Survived', data=df)


sns.countplot(x='Survived', data=df, hue='Sex')

sns.heatmap(df.isna())
(df['Age'].isna().sum()/len(df['Age']))*100

(df['Cabin'].isna().sum()/len(df['Age']))*100

sns.displot(x='Age',data=df)

df['Age'].fillna(df['Age'].mean(),inplace = True)

sns.heatmap(df.isna())

df.drop('Cabin', axis=1, inplace = True)
df.head()


df.info()
gender=pd.get_dummies(df['Sex'],drop_first=True)

df['Gender'] = gender


df.drop(['Name','Sex','Ticket','Embarked'],axis=1,inplace=True)

x=df[['PassengerId','Pclass','Age','SibSp','Parch','Fare','Gender']]
y=df['Survived']

# Data Modelling
# Building Model using Logestic Regression

# Build the model

# Training the modelling
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)

# Testing the model
from sklearn.metrics import confusion_matrix
pd.DataFrame(confusion_matrix(y_test,predict),columns=['Predicted No','Predicted Yes'],index=['Actual No','Actual Yes'])

#import classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,predict))
