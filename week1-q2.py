import pandas
import re
import numpy as np

from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv("C:\\Users\\Vladislav\\Desktop\\Machine Learning\\titanic.csv", index_col="PassengerId")

data1 = data.loc[ :,['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]

data2 = data1.dropna()

SexN = data2.Sex.factorize()

data2['SexN'] = SexN[0]

X = data2.loc[ :,['Pclass', 'Fare', 'Age', 'SexN']]
y = data2['Survived']

#X = np.array([[1, 2], [3, 4], [5, 6]])
#y = np.array([0, 1, 0])

clf = DecisionTreeClassifier()
clf.random_state = 241

clf.fit(X, y)

importances = clf.feature_importances_