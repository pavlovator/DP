import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import *
from torch.utils.data import DataLoader


class CNN_RV1(nn.Module):
    def __init__(self):
        super(CNN_RV1, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # in_channels, out_channels, kernel_size
        self.pool1 = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 3 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        # input shape: (batch_size, channels, height, width) : (b, 3, 100, 177)
        x = self.pool1(F.relu(self.conv1(x)))  # out: (N, 64, 6, 24, 43)
        x = self.pool2(F.relu(self.conv2(x)))  # out: (N, 64, 16, 10, 19)
        x = self.pool3(F.relu(self.conv3(x)))  # out: (N, 64, 32, 3, 7)
        x = x.view(-1, 32 * 3 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


def train(verbose_interval=1):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    train_loss, train_MAE = 0.0, 0.0
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print(labels.shape, outputs.shape, type(outputs))
        if epoch % verbose_interval == 0:
            outputs, val_loss = validation(criterion)
            print('epoch: {:} | train_loss: {:.2f} | validation_loss: {:.2f}'.format(epoch + 1, train_loss, val_loss))
            #update_history_RV1(history, epoch, train_loss, val_loss, t_mae, v_mae):
        train_loss, train_MAE = 0.0, 0.0


def validation(criterion):
    with torch.no_grad():
        net.train(False)
        inputs, labels = val_set[:]
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        net.train(True)
        return outputs, loss.item() / len(inputs)


def test():
    ...


if torch.cuda.is_available():
    dev = "cuda:0"
    print('Running on GPU')
else:
    dev = "cpu"
    print('Running on CPU')

device = torch.device(dev)

batch_size = 16
num_epochs = 200
train_val_splits = [214, 100]  # lengths of splits to be produced
history = {"epoch": [], "train_loss": [], "validation_loss": [], "train_MAE": [], "validation_MAE": []}


train_set, val_set = torch.utils.data.random_split(ScaledDataset("uniform_test/*", width=100), train_val_splits)

train_loader = DataLoader(dataset=train_set,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2)

net = CNN_RV1()
net.to(device)
train(verbose_interval=5)

#test_set = ScaledDataset("uniform_test/*", width=100)
