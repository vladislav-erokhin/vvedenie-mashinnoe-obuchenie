import numpy as np
import pandas
import heapq
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV

# читаем данные
newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )

# разделяем на признаки и результат
X = pandas.DataFrame(newsgroups.data)
y = pandas.DataFrame(newsgroups.target)

# трансформация tfidf
vectorizer = TfidfVectorizer(min_df=1)
# матрица, где каждому столбцу соотвествует слово
X1 = vectorizer.fit_transform(X[0])

# имена фич-столбцов = слова
fn = vectorizer.get_feature_names()

# обучение с разными параметрами C
grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X1, y[0])

# вывод значений
for a in gs.grid_scores_:
    # a.mean_validation_score — оценка качества по кросс-валидации
    print(a.mean_validation_score, " - ", a.parameters)
    # a.parameters — значения параметров

# победил C=1
clf1 = SVC(kernel='linear', random_state=241, C=1.0)
clf1.fit(X1, y[0])

c = clf1.coef_.toarray()

# считаем абсолютные значения весов для слов
cabs = [abs(number) for number in c]

# получаем топ 10 слов и их позиции
top10 = heapq.nlargest(10, enumerate(cabs[0]), key=lambda x: x[1])

# формируем список слов
words = list()
for e in top10:
    words.append(fn[e[0]])

# сортируем список слов
words.sort()

# выводим через запятую
result = ','.join(words)

print(result)
