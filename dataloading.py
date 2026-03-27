import torch
import numpy as np  
from torch.utils.data import Dataset, DataLoader

class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy =  np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[: , :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len
    

dataset = DiabetesDataset('diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)


        


class Model(torch.nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.linear1 = torch.nn.Linear(8,6)
    self.linear2 = torch.nn.Linear(6,4)
    self.linear3 = torch.nn.Linear(4,1)
    self.sigmoid = torch.nn.Sigmoid()

  def forward(self, x):
    x = self.sigmoid(self.linear1(x))
    x = self.sigmoid(self.linear2(x))
    x = self.sigmoid(self.linear3(x))
    return x

model = Model()

ceriterion = torch.nn.BCELoss(size_average = False)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)



if __name__ == "__main__":
    for epoch in range(1000):
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data

            # forward
            y_pre = model(inputs)
            loss = ceriterion(y_pre, labels)
            print(epoch, i, loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()

            # update parameter
            optimizer.step()