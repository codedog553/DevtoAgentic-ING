import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([2, 4, 6, 8, 10])

def forward(x: float, w: float) -> float:
    return w * x

def loss(x: float, y: float, w: float) -> float:
    y_pred = forward(x, w)
    return (y_pred - y) ** 2

w_list = []
mse_list = []

for w in np.arange(0, 4.1, 0.1):
    print(f"w = {w}")
    mse = 0
    for x, y in zip(x_data, y_data):
        y_pred = forward(x, w)
        mse += loss(x, y, w)
        print('\t', x, y, y_pred, loss(x, y, w))
    mse /= len(x_data)
    print(f"mse = {mse}")
    w_list.append(w)
    mse_list.append(mse)

plt.plot(w_list, mse_list)
plt.xlabel('w')
plt.ylabel('loss')
plt.title('loss vs w')
plt.show()