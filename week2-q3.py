import numpy
import pandas
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_boston
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

#sklearn.metrics.accuracy_score
#sklearn.preprocessing.StandardScaler

data_train = pandas.read_csv("C:\\Users\\Vladislav\\Desktop\\Machine Learning\\week2-q3\\perceptron-train.csv", header=None)
data_test = pandas.read_csv("C:\\Users\\Vladislav\\Desktop\\Machine Learning\\week2-q3\\perceptron-test.csv", header=None)

y_train = data_train[0]
X_train = data_train.loc[:, 1:2]

y_test = data_test[0]
X_test = data_test.loc[:, 1:2]

clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

score = metrics.accuracy_score(y_test, y_pred)

print(score)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = Perceptron(random_state=241)
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)

score1 = metrics.accuracy_score(y_test, y_pred)

print(score1)

print(score1 - score)