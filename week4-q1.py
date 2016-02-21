import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge

data_train = pd.read_csv("week4-q1\\salary-train.csv")
data_test = pd.read_csv("week4-q1\\salary-test-mini.csv ")

data_train['FullDescription'] = data_train['FullDescription'].str.lower()
data_train['FullDescription'] = data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)

data_test['FullDescription'] = data_test['FullDescription'].str.lower()
data_test['FullDescription'] = data_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
data_test['LocationNormalized'].fillna('nan', inplace=True)
data_test['ContractTime'].fillna('nan', inplace=True)

vectorizer = TfidfVectorizer(min_df=5)
X_train_fd_vect = vectorizer.fit_transform(data_train['FullDescription'])
X_test_fd_vect = vectorizer.transform(data_test['FullDescription'])

enc = DictVectorizer()
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

X_train = hstack([X_train_fd_vect, X_train_categ])
X_test = hstack([X_test_fd_vect, X_test_categ])

r = Ridge(alpha=1)
r.fit(X_train, data_train['SalaryNormalized'])

p = r.predict(X_test)

print("Predicted salary: ", " ".join(map(str, np.round(p, 2))))