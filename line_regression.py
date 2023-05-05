import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-5, 5, num=100)[:, None]
y = -0.5 + 2.2 * X + 0.3 * X ** 3 + 2 * np.random.randn(100, 1)

plt.plot(X, y)

plt.show()

# задаём начальные случайные значения коэффициентам линейной регрессии
a = np.random.randn(1)
b = np.random.randn(1)
c = np.random.randn(1)
d = np.random.randn(1)
print(a, b, c, d)

X_2 = [x ** 2 for x in X]
X_3 = [x ** 3 for x in X]


for i in range(1000):
    yhat = a + b * X + c * X_2 + d * X_3 # Задаю полином

    error = (y - yhat) # Просчитываю прогрешность

    a_grad = - 1 / len(X) * error.mean() # берем производные по a, b ,c

    b_grad = - 1 / len(X) * (X * error).mean()

    c_grad = - 1 / len(X) * (X_2 * error).mean()

    d_grad = - 1 / len(X) * (X_3 * error).mean()

    a = a - 0.01 * a_grad
    b = b - 0.01 * b_grad
    c = c - 0.01 * c_grad
    d = d - 0.01 * d_grad

plt.plot(X, y)
plt.plot(X, yhat, color='red')

plt.show()
print(a, b, c, d)
