import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0, 4.0, 5.0]
y_data = [2.0, 4.0, 6.0, 8.0, 10.0]

w = 0.00001
lr = 0.001          # 学习率

def forward(x):
    return w * x

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

def gradient(x, y):
    return 2 * x * (x * w - y)

epoch_list = []
loss_list = []

print('Predict before training: f(5) =', forward(5))

for epoch in range(100):
    epoch_loss = 0.0
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w -= lr * grad                 # 正确更新
        epoch_loss += loss(x, y)
    epoch_loss /= len(x_data)           # 平均损失
    epoch_list.append(epoch)
    loss_list.append(epoch_loss)
    print(f"progress: {epoch} w={w:.6f} loss={epoch_loss:.6f}")

print('Predict after training: f(5) =', forward(5))

plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('loss vs epoch')
plt.show()


import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
x_data = np.arange(1, 6, dtype=float)
y_data = 2 * x_data + np.random.randn(5) * 0.5  # 加入噪声

w = 0.1
lr = 0.00001

def forward(x): return w * x
def loss(x, y): return (forward(x) - y) ** 2
def gradient(x, y): return 2 * x * (w * x - y)

loss_list = []
iter_list = []

for epoch in range(30):
    # 随机打乱数据
    indices = np.random.permutation(len(x_data))
    for i in indices:
        x, y = x_data[i], y_data[i]
        grad = gradient(x, y)
        w -= lr * grad
        loss_list.append(loss(x, y))
        iter_list.append(len(iter_list))

plt.plot(iter_list, loss_list)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('SGD with noisy data')
plt.show()


'''
在机器学习中，**\( w \)** 和 **\( lr \)（学习率）** 是模型训练中两个最基础也最重要的概念。

---

## 1. 权重 \( w \) —— 模型要学习的参数

### 作用
- **定义模型的行为**：在线性回归 \( y = w \cdot x \) 中，\( w \) 就是斜率，它决定了输入 \( x \) 如何影响输出 \( y \)。例如 \( w = 2 \) 时，\( x \) 每增加 1，\( y \) 就增加 2。
- **存储知识**：训练完成后，\( w \) 的值就是模型从数据中学到的“知识”。对于更复杂的模型（如神经网络），\( w \) 是连接神经元之间的权重，控制信息流动的强弱。

### 意义
- \( w \) 是**可训练的参数**，它的值不是我们事先设定的，而是通过优化算法（如梯度下降）从数据中自动调整出来的。
- 初始的 \( w \) 通常随机设置（如很小的值），然后通过不断计算梯度来更新，最终收敛到一个能使损失最小的值。

---

## 2. 学习率 \( lr \) —— 控制参数更新步长的超参数

### 作用
- **决定每次参数更新的幅度**：梯度下降的更新公式是  
  \[
  w_{\text{new}} = w_{\text{old}} - lr \cdot \text{梯度}
  \]  
  其中梯度指明了损失函数下降最快的方向，而 \( lr \) 则控制我们沿着这个方向走多远。

### 意义
- **平衡收敛速度与稳定性**：
  - 如果 \( lr \) **太大**：参数更新步长过大，可能会“跳过”最优点，导致损失震荡甚至发散（你之前遇到的梯度爆炸就是变相的步长过大）。
  - 如果 \( lr \) **太小**：更新缓慢，需要很多轮迭代才能收敛，训练时间变长，还可能陷入局部最优。
- **是超参数**：\( lr \) 不是模型学出来的，而是训练前由人工设定的。通常需要根据数据和模型进行调整，有时还会在训练过程中动态改变（如学习率衰减）。

---

## 3. \( w \) 与 \( lr \) 的关系

- **\( w \) 是目标**：我们最终想要一个合适的 \( w \) 让模型预测准确。
- **\( lr \) 是手段**：它控制我们如何一步步逼近这个目标。没有 \( lr \)，梯度只告诉我们方向，却不知道迈多大的步子。

### 为什么需要独立的学习率？
从你的代码错误 `w -= w * grad` 就能看出学习率独立的重要性：  
你无意中把当前权重 \( w \) 当成了学习率，导致步长随 \( w \) 变化——当 \( w \) 很小时步长小，但当 \( w \) 变大后步长急剧增大，最终失控。  
正确的做法是用一个**独立且固定（或按计划变化）的 \( lr \)**，让更新步长可控。

---

## 4. 实际应用中的注意事项

- **权重初始化**：\( w \) 的初始值会影响训练。如果初始值离最优点太远，可能需要更多时间；如果初始值过大，可能导致梯度爆炸。
- **学习率的选择**：通常从 0.01、0.001 等小值开始尝试。对于不同问题，最优学习率差异很大。现代优化器（如 Adam）会自适应调整每个参数的学习率，但仍需设置一个初始全局学习率。
- **学习率调度**：在训练过程中逐步减小学习率，可以在初期快速下降，后期精细收敛。

---

## 总结

| 概念 | 角色 | 是否可训练 | 典型取值 | 影响 |
|------|------|------------|----------|------|
| **权重 \( w \)** | 模型的核心参数，决定预测值 | 是（通过梯度下降更新） | 由数据决定（如 2.0） | 直接影响模型输出 |
| **学习率 \( lr \)** | 控制参数更新步长 | 否（超参数，需手动设定） | 0.01, 0.001 等 | 影响收敛速度和稳定性 |

'''