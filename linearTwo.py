import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([2, 4, 6, 8, 10])

def forward(x, w, b):
    return w * x + b

def loss(x, y, w, b):
    y_pred = forward(x, w, b)
    return (y_pred - y) ** 2

# 定义 w 和 b 的取值区间
w_range = np.arange(0, 4.1, 0.1)
b_range = np.arange(0, 4.1, 0.1)

# 准备存储 MSE 的二维数组
mse_matrix = np.zeros((len(w_range), len(b_range)))

# 双重循环计算每个 (w, b) 组合的 MSE
for i, w in enumerate(w_range):
    for j, b in enumerate(b_range):
        # 计算当前 (w,b) 下的总损失
        total_loss = 0
        for x, y in zip(x_data, y_data):
            total_loss += loss(x, y, w, b)
        mse = total_loss / len(x_data)
        mse_matrix[i, j] = mse  # 注意：这里 mse_matrix[i, j] 对应 w_range[i], b_range[j]

# 生成网格坐标矩阵
W, B = np.meshgrid(w_range, b_range, indexing='ij')  # 使用 indexing='ij' 使 W 的行对应 w，列对应 b

# 创建3D图形
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面（注意此时 mse_matrix 的形状与 W 一致，无需转置）
surf = ax.plot_surface(W, B, mse_matrix, cmap='viridis', edgecolor='none')

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='MSE')
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('MSE')
ax.set_title('Loss Surface (3D)')

plt.show()