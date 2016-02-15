import pandas
from sklearn.svm import SVC

svm_data = pandas.read_csv("C:\\Users\\Vladislav\\Desktop\\Machine Learning\\week3-q1\\svm-data.csv", header=None)

y = svm_data[0]
X = svm_data.loc[:, 1:2]

clf = SVC(C=100000, random_state=241, kernel='linear')
clf.fit(X, y)

print(clf.support_vectors_)
print(clf.support_)
