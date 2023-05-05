import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

# Подгружаем датасет
url = "iris.data"
# Assign colum names to the dataset names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class'] # Read dataset to pandas dataframe dataset = pd.read_csv(url, names=names)
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
# Read dataset to pandas dataframe dataset = pd.read_csv(url, names=names)
dataset = pd.read_csv(url, names=names)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Разбиваем выборку на обучающую и тестовую
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Объявляем переменные для каждого параметра каждого ядра, и количество эллементов ядра

X1_for_core1 = 0
X2_for_core1 = 0
X3_for_core1 = 0
X4_for_core1 = 0
n_core1 = 0

X1_for_core2 = 0
X2_for_core2 = 0
X3_for_core2 = 0
X4_for_core2 = 0
n_core2 = 0

X1_for_core3 = 0
X2_for_core3 = 0
X3_for_core3 = 0
X4_for_core3 = 0
n_core3 = 0

# Выбираем три случайных эллемента

core3 = X_train[1]
core2 = X_train[len(X_train) - 1]
core1 = X_train[(int)(len(X_train) / 2)]

for z in range(0, 10):
    # Обнуляем массив у
    y_pred_me = []
    for i in range(0, len(X_train)):

        # Считаем расстояние каждого эллемента до каждого ядра
        length1 = math.sqrt(((X_train[i][0] - core1[0]) ** 2) + ((X_train[i][1] - core1[1]) ** 2) + (
                    (X_train[i][2] - core1[2]) ** 2) + ((X_train[i][3] - core1[3]) ** 2))
        length2 = math.sqrt(((X_train[i][0] - core2[0]) ** 2) + ((X_train[i][1] - core2[1]) ** 2) + (
                    (X_train[i][2] - core2[2]) ** 2) + ((X_train[i][3] - core2[3]) ** 2))
        length3 = math.sqrt(((X_train[i][0] - core3[0]) ** 2) + ((X_train[i][1] - core3[1]) ** 2) + (
                    (X_train[i][2] - core3[2]) ** 2) + ((X_train[i][3] - core3[3]) ** 2))

        # Определяем к какому ядру отнести эллемент
        if length1 < length2:
            if length1 < length3:
                y_pred_me.append(0)

        if length2 < length1:
            if length2 < length3:
                y_pred_me.append(1)

        if length3 < length2:
            if length3 < length1:
                y_pred_me.append(2)

                # После этого пересчитываем ядра
    for j in range(0, len(X_train)):
        if y_pred_me[j] == 0:
            X1_for_core1 += X_train[j][0]
            X2_for_core1 += X_train[j][1]
            X3_for_core1 += X_train[j][2]
            X4_for_core1 += X_train[j][3]
            n_core1 += 1
        if y_pred_me[j] == 1:
            X1_for_core2 += X_train[j][0]
            X2_for_core2 += X_train[j][1]
            X3_for_core2 += X_train[j][2]
            X4_for_core2 += X_train[j][3]
            n_core2 += 1
        if y_pred_me[j] == 2:
            X1_for_core3 += X_train[j][0]
            X2_for_core3 += X_train[j][1]
            X3_for_core3 += X_train[j][2]
            X4_for_core3 += X_train[j][3]
            n_core3 += 1
    core1 = [X1_for_core1 / n_core1, X2_for_core1 / n_core1, X3_for_core1 / n_core1, X4_for_core1 / n_core1]
    core2 = [X1_for_core2 / n_core2, X2_for_core2 / n_core2, X3_for_core2 / n_core2, X4_for_core2 / n_core2]
    core3 = [X1_for_core3 / n_core3, X2_for_core3 / n_core3, X3_for_core3 / n_core3, X4_for_core3 / n_core3]

# Визуализируем для исходных данных
fig = plt.figure()
ax = fig.add_subplot()

plt.scatter(X_train[y_train == 'Iris-setosa', 0], X_train[y_train == 'Iris-setosa', 1], s=70, c='purple',
            label='Iris-setosa')
plt.scatter(X_train[y_train == 'Iris-versicolor', 0], X_train[y_train == 'Iris-versicolor', 1], s=70, c='orange',
            label='Iris-versicolour')
plt.scatter(X_train[y_train == 'Iris-virginica', 0], X_train[y_train == 'Iris-virginica', 1], s=70, c='green',
            label='Iris-virginica')
# Plotting the centroids of the clusters
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'red', label = 'Centroids')

plt.legend()
# plt.show()


# Визуализируем для наших вычислений
fig = plt.figure()
ax = fig.add_subplot()
for i in range(1, len(y_pred_me)):
    if y_pred_me[i] == 0:
        plt.scatter(X_train[i][0], X_train[i][1], s=70, c='purple')
    if y_pred_me[i] == 1:
        plt.scatter(X_train[i][0], X_train[i][1], s=70, c='orange')
    if y_pred_me[i] == 2:
        plt.scatter(X_train[i][0], X_train[i][1], s=70, c='green')

plt.legend()
plt.show()



