import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ==================== 1. 数据预处理 ====================

column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
categorical_cols = [c for c in column_names if c not in numerical_cols and c != 'income']

# ---------- 训练集 ----------
train_df = pd.read_csv('train.txt', header=None, names=column_names, na_values='?')

# 清洗 income 列：去除前后空格，过滤无效标签
train_df['income'] = train_df['income'].str.strip()
train_df = train_df[train_df['income'].isin(['<=50K', '>50K'])]
print("训练集有效样本数:", len(train_df))

# 数值列转数值类型并填充缺失值（中位数）
for col in numerical_cols:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
train_df[numerical_cols] = train_df[numerical_cols].fillna(train_df[numerical_cols].median())

# 类别列填充缺失值
for col in categorical_cols:
    train_df[col] = train_df[col].fillna('unknown')

# 对 capital-gain 和 capital-loss 进行对数变换（压缩极端值）
train_df['capital-gain'] = np.log1p(train_df['capital-gain'])
train_df['capital-loss'] = np.log1p(train_df['capital-loss'])

# 标签映射
y_train = train_df['income'].map({'<=50K': -1, '>50K': 1}).values

# 标准化数值特征
scaler = StandardScaler()
train_numerical = scaler.fit_transform(train_df[numerical_cols])

# 独热编码类别特征
train_categorical = pd.get_dummies(train_df[categorical_cols])
train_cat_cols = train_categorical.columns

# 合并特征
X_train = np.concatenate([train_numerical, train_categorical.values], axis=1)

# ---------- 测试集 ----------
# 读取测试集特征
test_df = pd.read_csv('test.txt', header=None, names=column_names[:-1], na_values='?')

# 读取 groundtruth 并清洗
truth_df = pd.read_csv('test_ground_truth.txt', header=None, names=column_names, na_values='?')
truth_df['income'] = truth_df['income'].str.strip().str.replace('.', '', regex=False)
truth_df = truth_df[truth_df['income'].isin(['<=50K', '>50K'])]

# 对齐特征与标签（假设顺序一致）
test_df = test_df.iloc[truth_df.index].reset_index(drop=True)
truth_df = truth_df.reset_index(drop=True)

# 提取标签
y_test_true = truth_df['income'].map({'<=50K': -1, '>50K': 1}).values

# 数值列清洗
for col in numerical_cols:
    test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
for col in numerical_cols:
    test_df[col] = test_df[col].fillna(train_df[col].median())   # 用训练集的中位数填充

# 类别列填充
for col in categorical_cols:
    test_df[col] = test_df[col].fillna('unknown')

# 对数变换
test_df['capital-gain'] = np.log1p(test_df['capital-gain'])
test_df['capital-loss'] = np.log1p(test_df['capital-loss'])

# 标准化（使用训练集的 scaler）
test_numerical = scaler.transform(test_df[numerical_cols])

# 独热编码
test_categorical = pd.get_dummies(test_df[categorical_cols])

# 对齐训练集的类别列（补全缺失列，删除多余列）
for col in train_cat_cols:
    if col not in test_categorical.columns:
        test_categorical[col] = 0
test_categorical = test_categorical[train_cat_cols]

# 合并特征
X_test = np.concatenate([test_numerical, test_categorical.values], axis=1)

# ---------- 转换为 PyTorch Tensor ----------
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test_true = torch.tensor(y_test_true, dtype=torch.float32)

# 检查数据是否含有 NaN 或 inf
print("X_train has NaN:", torch.isnan(X_train).any().item())
print("X_train has inf:", torch.isinf(X_train).any().item())
print("y_train has NaN:", torch.isnan(y_train).any().item())
print("X_test has NaN:", torch.isnan(X_test).any().item())
print("y_test has NaN:", torch.isnan(y_test_true).any().item())

# 创建 DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)   # mini-batch SGD

test_dataset = TensorDataset(X_test, y_test_true)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print("训练集特征维度:", X_train.shape[1])
print("测试集特征维度:", X_test.shape[1])
print("训练集样本数:", len(train_dataset))
print("测试集样本数:", len(test_dataset))

# ==================== 2. 模型定义 ====================

class LinearSVM(nn.Module):
    def __init__(self, input_dim):
        super(LinearSVM, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        return self.linear(x).squeeze()

def hinge_loss(outputs, labels):
    return torch.mean(torch.clamp(1 - labels * outputs, min=0))

def total_loss(model, outputs, labels, C):
    l2_reg = 0.5 * torch.sum(model.linear.weight ** 2)
    hinge = hinge_loss(outputs, labels)
    return l2_reg + C * hinge

# ==================== 3. 训练与评估函数 ====================

def train_model(model, train_loader, C, epochs, lr):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss_epoch = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = total_loss(model, outputs, batch_y, C)
            # 数值稳定性检查
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf loss at epoch {epoch+1}, exiting")
                return
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss_epoch += loss.item()
        avg_loss = total_loss_epoch / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            preds = torch.sign(outputs)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    return correct / total

# ==================== 4. 实验不同 C 值 ====================

input_dim = X_train.shape[1]
C_values = [1e-3, 1e-2, 1e-1, 1]
results = {}

for C in C_values:
    print(f"\nTraining with C = {C}")
    model = LinearSVM(input_dim)
    train_model(model, train_loader, C, epochs=20, lr=0.0001)
    acc = evaluate(model, test_loader)
    results[C] = acc
    print(f"Test accuracy: {acc:.4f}")

print("\nResults:")
for C, acc in results.items():
    print(f"C = {C:.0e} -> Accuracy = {acc:.4f}")