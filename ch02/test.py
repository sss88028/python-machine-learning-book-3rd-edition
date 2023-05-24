import matplotlib.pyplot as plt
import pandas as pd
import requests
import io
import os
from perceptron import Perceptron
PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

with requests.get(PATH) as response:
    raw_data = response.text

# s = os.path.join("https://archive.ics.uci.edu", "ml",
#                  "machine-learning-databases",
#                  "iris", "iris.data")
s = os.path.join(PATH)

df = pd.read_csv(s)
# print(df.tail())
# string_data = io.StringIO(raw_data)
#
# df = pd.read_csv(string_data, header=None, encoding="utf-8")
#
# print(df.tail())

import matplotlib.pyplot as plot
import numpy as np

y = df.iloc[:, 4].values
y = np.where(y == "Iris-setosa", -1, 1)

X = df.iloc[:, [0, 2, 4]].values
setosa_array = X[(X[:, 2] == "Iris-setosa")]
virginica_array = X[(X[:, 2] == "Iris-virginica")]
versicolor_array = X[(X[:, 2] == "Iris-versicolor")]

plt.scatter(setosa_array[:, 0], setosa_array[:, 1],
            color="red", marker="o", label="setosa")
plt.scatter(virginica_array[:, 0], virginica_array[:, 1],
            color="blue", marker="x", label="virginica")
plt.scatter(versicolor_array[:, 0], versicolor_array[:, 1],
            color="green", marker="^", label="versicolor")

plt.xlabel("sepal length [cm]")
plt.ylabel("petal length [cm]")
plt.legend(loc="upper left")
plt.show()

ppn = Perceptron(eta=0.1, n_iter=10)
# ppn.fit(X, y)