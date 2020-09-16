import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from utils import *


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.h1 = nn.Linear(28*28,300)
        self.fc1 = nn.Tanh()
        self.h2 = nn.Linear(300,100)
        self.fc2 = nn.Tanh()
        self.out = nn.Linear(100,10)
        self.fc3 = nn.Sigmoid()

    def forward(self, x):
        x = self.fc3(self.out(self.fc2(self.h2(self.fc1(self.h1(x))))))
        return x


batch_size = 5000
n_epochs = 10


train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size, shuffle=True)




def train():
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.view(data.shape[0], -1)
        target = torch.nn.functional.one_hot(target, num_classes=10).float()
        data, target = data.to(device), target.to(device)
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

if not torch.cuda.is_available():
    dev = "cuda:0"
    print('Running on GPU')
else:
    dev = "cpu"
    print('Running on CPU')

device = torch.device(dev)


network = Net()
criterion = F.mse_loss
optimizer = optim.Adam(network.parameters(), lr=0.1)
network.to(device)


tic = time.perf_counter()
times = []
for epoch in range(1, n_epochs + 1):
    train()
    toc = time.perf_counter()
    print("epoch: {:} time: {:.2f} seconds".format(epoch, toc - tic))
    times.append(str(round(toc - tic, 2)))
    print(get_gpu_memory_map())

save_times(times, "test1_GPU.txt")