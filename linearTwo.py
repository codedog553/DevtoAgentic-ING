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



'''
`np.meshgrid` 是 NumPy 中一个非常实用的函数，主要用于从一维坐标数组生成二维网格坐标矩阵，常见于可视化（如等高线图、3D 曲面图）和向量化计算。下面我会详细介绍它的用法、参数、常见问题以及实际应用。

---

## 1. `np.meshgrid` 的基本作用

假设我们有一组 \(x\) 坐标值 `x = [1, 2, 3]` 和一组 \(y\) 坐标值 `y = [4, 5, 6]`。我们想要在 \(x\)-\(y\) 平面上形成一个网格，每个网格点由一对 \((x_i, y_j)\) 组成。`meshgrid` 的作用就是生成两个二维数组 `X` 和 `Y`，使得：
- `X[i, j] = x[j]`（默认情况下）
- `Y[i, j] = y[i]`

这样，`(X[i,j], Y[i,j])` 就对应了网格上的一个点。

### 示例
```python
import numpy as np

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

X, Y = np.meshgrid(x, y)
print("X:\n", X)
print("Y:\n", Y)
```
输出：
```
X:
 [[1 2 3]
 [1 2 3]
 [1 2 3]]
Y:
 [[4 4 4]
 [5 5 5]
 [6 6 6]]
```
可以看到：
- `X` 的每一行都是 `x` 的复制（行数等于 `y` 的长度）
- `Y` 的每一列都是 `y` 的复制（列数等于 `x` 的长度）

---

## 2. 参数详解

### 2.1 `indexing` 参数
`indexing` 有两个可选值：`'xy'`（默认）和 `'ij'`。它们决定了输出的坐标顺序。

- **`indexing='xy'`（默认）**  
  此时，输出的第一个数组 `X` 的行对应 `y` 坐标，列对应 `x` 坐标；第二个数组 `Y` 的行对应 `y`，列对应 `x`。这种约定与笛卡尔坐标系一致，常用于绘图（如 `matplotlib` 中，`x` 是横轴，`y` 是纵轴）。

- **`indexing='ij'`**  
  此时，输出的第一个数组 `X` 的行对应 `x` 坐标，列对应 `y` 坐标；第二个数组 `Y` 的行对应 `x`，列对应 `y`。这种约定与矩阵索引（先行后列）一致。

#### 对比示例
```python
X_xy, Y_xy = np.meshgrid(x, y, indexing='xy')
print("indexing='xy':")
print("X shape:", X_xy.shape)  # (3, 3) 即 (len(y), len(x))
print("Y shape:", Y_xy.shape)  # (3, 3)

X_ij, Y_ij = np.meshgrid(x, y, indexing='ij')
print("indexing='ij':")
print("X shape:", X_ij.shape)  # (3, 3) 即 (len(x), len(y))
print("Y shape:", Y_ij.shape)  # (3, 3)
```

**形状总结**：
- `indexing='xy'`：输出形状为 `(len(y), len(x))`
- `indexing='ij'`：输出形状为 `(len(x), len(y))`

### 2.2 `sparse` 参数
当 `sparse=True` 时，输出的是稀疏网格（即返回的数组维度会减少，节省内存），但通常用于广播计算。默认 `sparse=False`，返回完整网格。

```python
X_sp, Y_sp = np.meshgrid(x, y, sparse=True)
print("X_sp shape:", X_sp.shape)  # (1, 3)
print("Y_sp shape:", Y_sp.shape)  # (3, 1)
```
稀疏网格在进行向量化运算时非常高效，因为广播机制会自动扩展。

---

## 3. 在绘图中的应用

### 3.1 等高线图 `contourf`
`plt.contourf(X, Y, Z)` 要求：
- `X` 和 `Y` 可以是二维网格坐标矩阵（由 `meshgrid` 生成），也可以是长度分别为 `n` 和 `m` 的一维数组（此时函数会自动广播成网格）。
- `Z` 必须是二维数组，形状与 `X`、`Y` 一致。

**注意**：如果你使用 `meshgrid` 生成 `X` 和 `Y`，一定要注意 `indexing` 的设置，使得 `Z` 的索引与坐标对应。

#### 示例：绘制损失曲面
沿用之前线性回归的例子：
```python
w = np.arange(0, 4.1, 0.1)   # 对应 x 轴
b = np.arange(0, 4.1, 0.1)   # 对应 y 轴
# 假设已经计算出 mse_matrix，形状为 (len(w), len(b))，即 mse_matrix[i,j] 对应 (w[i], b[j])

# 方法1：使用 indexing='ij' 生成网格，直接与 mse_matrix 对应
W, B = np.meshgrid(w, b, indexing='ij')
plt.contourf(W, B, mse_matrix, levels=50, cmap='viridis')
plt.xlabel('w')
plt.ylabel('b')
plt.colorbar()
plt.show()

# 方法2：使用默认 indexing='xy'，此时需要转置 mse_matrix
W, B = np.meshgrid(w, b, indexing='xy')  # 默认
plt.contourf(W, B, mse_matrix.T, levels=50, cmap='viridis')  # 注意转置
```
为什么需要转置？因为默认 `indexing='xy'` 生成的 `W` 形状是 `(len(b), len(w))`，即行对应 `b`，列对应 `w`，而我们的 `mse_matrix` 是 `(len(w), len(b))`，所以要用 `.T` 将行列互换。

### 3.2 3D 曲面图 `plot_surface`
与 `contourf` 类似，`plot_surface` 也要求 `X`、`Y`、`Z` 形状相同。推荐使用 `indexing='ij'` 生成网格，避免转置的麻烦。

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
W, B = np.meshgrid(w, b, indexing='ij')
ax.plot_surface(W, B, mse_matrix, cmap='viridis')
```

---

## 4. 在向量化计算中的应用

`meshgrid` 最常见的用途之一是将循环计算转化为向量化运算。例如，计算函数 \(f(w,b) = w^2 + b^2\) 在所有网格点上的值，可以这样写：

```python
w = np.arange(0, 5, 1)
b = np.arange(0, 5, 1)
W, B = np.meshgrid(w, b, indexing='ij')
Z = W**2 + B**2   # 直接对二维数组进行运算，比双重循环快得多
```

---

## 5. 常见问题与注意事项

- **形状不匹配**：如果 `Z` 的形状与 `X`、`Y` 不一致，绘图会出错。务必检查形状。
- **转置困惑**：很多初学者在使用默认 `indexing` 时忘记转置，导致图像颠倒。可以设置 `indexing='ij'` 来符合矩阵索引习惯。
- **多个数组**：`meshgrid` 可以接受多于两个一维数组，生成对应维度的网格。例如 `np.meshgrid(x, y, z)` 会生成三个三维数组，用于三维空间网格。

---

## 6. 总结

- `np.meshgrid` 将一维坐标数组转换为二维网格坐标矩阵。
- 通过 `indexing` 参数控制输出形状：`'xy'`（默认，适合绘图）或 `'ij'`（适合矩阵索引）。
- 在绘图时，确保 `Z` 的形状与网格矩阵匹配，必要时转置或调整 `indexing`。
- 利用 `meshgrid` 可以实现高效的向量化计算，避免显式循环。

掌握 `meshgrid` 是进行科学计算和可视化的基础技能，希望这篇讲解对你有所帮助！
'''