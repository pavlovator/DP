import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import ScaledDataset
from torch.utils.data import DataLoader


class CNN_V1(nn.Module):
    def __init__(self):
        super(CNN_V1, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) #in_channels, out_channels, kernel_size
        self.pool1 = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(8, 8)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(net, train_loader):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    running_loss = 0.0
    for epoch in range(2):
        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 3 == 2:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 3))
                running_loss = 0.0




dataset = ScaledDataset("new_pics/*", 100)
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)

#net = CNN_V1()
#train(net, train_loader)