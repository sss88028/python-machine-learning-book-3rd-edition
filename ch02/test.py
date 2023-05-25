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

y = df.iloc[:100, 4].values
y = np.where(y == "Iris-setosa", -1, 1)

X = df.iloc[:100, [0, 2]].values

# temp_X = df.iloc[:, [0, 2, 4]].values
# setosa_array = temp_X[(temp_X[:, 2] == "Iris-setosa")]
# virginica_array = temp_X[(temp_X[:, 2] == "Iris-virginica")]
# versicolor_array = temp_X[(temp_X[:, 2] == "Iris-versicolor")]
#
# plt.scatter(setosa_array[:, 0], setosa_array[:, 1],
#             color="red", marker="o", label="setosa")
# plt.scatter(virginica_array[:, 0], virginica_array[:, 1],
#             color="blue", marker="x", label="virginica")
# plt.scatter(versicolor_array[:, 0], versicolor_array[:, 1],
#             color="green", marker="^", label="versicolor")
#
# plt.xlabel("sepal length [cm]")
# plt.ylabel("petal length [cm]")
# plt.legend(loc="upper left")
# plt.show()

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
# plt.plot(range(1, len(ppn.error_) + 1),
#          ppn.error_, marker="o")
# plt.xlabel("Epochs")
# plt.xlabel("Number of updates")
# plt.show()

from matplotlib.colors import ListedColormap

# for idx, cl in enumerate(np.unique(y)):
#     print(f"index : {idx}, cl {cl}")
#     print(X[y == cl, 0])

def plot_decision_regions(X, y, classifier: Perceptron, resolution=0.02):
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    unique_len = len(np.unique(y))
    cmap = ListedColormap(colors[:unique_len])

    x1_min = X[:, 0].min() - 1
    x1_max = X[:, 0].max() + 1

    x2_min = X[:, 1].min() - 1
    x2_max = X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=cmap.colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolors="black")


plot_decision_regions(X, y, classifier=ppn)
plt.xlabel("sepal length [cm]")
plt.ylabel("petal length [cm]")
plt.legend(loc="upper left")
plt.show()