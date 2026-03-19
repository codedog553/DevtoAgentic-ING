from typing import List
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0, 4.0, 5.0]
y_data = [2.0, 4.0, 6.0, 8.0, 10.0]

w = 1.0

def forward(x: float) -> float:
    return w * x

def cost(xs, ys):
    cost = 0.0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)

def gradient(xs, ys):
    grad = 0.0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        grad += 2 * (y_pred - y) * x
    return grad / len(xs)

epoch_list = []
cost_list = []

print('Predict before training: f(5) =', forward(5))
for epoch in range(100):
    grad = gradient(x_data, y_data)
    w -= 0.01 * grad
    epoch_list.append(epoch)
    cost_list.append(cost(x_data, y_data))
    print(f'epoch: {epoch}, w: {w}, loss: {cost(x_data, y_data)}')

print('Predict after training: f(5) =', forward(5))

plt.plot(epoch_list, cost_list)
plt.xlabel('epoch')
plt.ylabel('cost')
plt.title('cost vs epoch')
plt.show()