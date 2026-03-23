import torch
import torch.nn.functional as F

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[0], [0], [1]], dtype=torch.float)

class LogisticModel(torch.nn.Module):
    def __init__(self):
        super(LogisticModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1) 

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x)) 
        return y_pred
    
model = LogisticModel()

criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w = ', model.linear.weight.item(), 'b = ', model.linear.bias.item())
x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 200)
x_t = torch.tensor(x).view(200, 1).float()
y_t = model(x_t)

y = y_t.data.numpy()
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Logistic Regression with PyTorch')
plt.grid()
plt.show()