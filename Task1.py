import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
titanic_data=pd.read_csv('C:\\Users\\Gauri Rakhunde\\Documents\\Data_Science\\pythonProject1\\Titanic-dataset.csv')
print(titanic_data.head(10))

sns.countplot(x="Survived",data=titanic_data)
plt.show()

sns.countplot(x="Survived",hue="Sex",data=titanic_data)
plt.show()

sns.countplot(x="Survived",hue="Pclass",data=titanic_data)
plt.show()

titanic_data["Age"].plot.hist()
plt.show()

titanic_data["Fare"].plot.hist()
plt.show()

sns.countplot(x="SibSp",data=titanic_data)
plt.show()


print(titanic_data.isnull().sum())

titanic_data.drop("Cabin", axis=1, inplace=True)
print(titanic_data.head(5))
titanic_data.dropna(inplace=True)
print(titanic_data.isnull().sum())

sex=pd.get_dummies(titanic_data['Sex'], drop_first=True)
embark=pd.get_dummies(titanic_data['Embarked'], drop_first=True)
Pcl=pd.get_dummies(titanic_data['Pclass'], drop_first=True)

titanic_data=pd.concat([titanic_data,sex,embark,Pcl],axis=1)

print(titanic_data.head(5))
titanic_data.drop(['Sex','Embarked','PassengerId','Name',"Ticket",'Pclass'],axis=1,inplace=True)
print(titanic_data.head(5))

x = titanic_data.drop("Survived", axis=1)
y = titanic_data["Survived"]

# Ensure all feature names are strings
x.columns = x.columns.astype(str)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Train the logistic regression model
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)

# Make predictions
predictions = logmodel.predict(x_test)


print(classification_report(y_test, predictions))
print(confusion_matrix(y_test,predictions))
print(accuracy_score(y_test,predictions))



