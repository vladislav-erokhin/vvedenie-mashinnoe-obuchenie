import pandas
import re
from collections import Counter

data = pandas.read_csv("C:\\Users\\Vladislav\\Desktop\\Machine Learning\\titanic.csv", index_col="PassengerId")

female = data[data["Sex"] == "female"]

namesLst = list()
for name in female["Name"]:
    nameArr = re.split(" |\.|\(|\)|,", name)
    for word in nameArr:
        namesLst.append(word)

c = Counter(namesLst)

p = pandas.Series(data = namesLst)