import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import torchvision.transforms as transforms
from utils import *


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.batch_norm1 = torch.nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.batch_norm2 = torch.nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.batch_norm3 = torch.nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, 5)
        self.batch_norm4 = torch.nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, 5)
        self.batch_norm5 = torch.nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 256, 5)
        self.batch_norm6 = torch.nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 5 * 5, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.pool(self.batch_norm1(F.relu(self.conv1(x))))
        x = self.pool(self.batch_norm2(F.relu(self.conv2(x))))
        x = self.batch_norm3(F.relu(self.conv3(x)))
        x = self.batch_norm4(F.relu(self.conv4(x)))
        x = self.batch_norm5(F.relu(self.conv5(x)))
        x = self.batch_norm6(F.relu(self.conv6(x)))
        x = x.view(-1, 256 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


batch_size = 512
n_epochs = 5


train_loader = torch.utils.data.DataLoader(torchvision.datasets.STL10(root="files/",
                                        download=True, transform=transforms.Compose(
                                                                [transforms.ToTensor(),
                                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                                        )), batch_size=batch_size, shuffle=True, num_workers=2)


def train():
    network.train()
    for batch_idx, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = network(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

if torch.cuda.is_available():
    dev = "cuda:0"
    print('Running on GPU')
else:
    dev = "cpu"
    print('Running on CPU')

device = torch.device(dev)


network = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr=0.001)
network.to(device)


tic = time.perf_counter()
times = []
for epoch in range(1, n_epochs + 1):
    train()
    toc = time.perf_counter()
    print("epoch: {:} time: {:.2f} seconds".format(epoch, toc - tic))
    times.append(str(round(toc - tic, 2)))
    print(get_gpu_memory_map())

save_times(times, "test2_GPU.txt")
