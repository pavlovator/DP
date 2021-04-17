import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import *
from torch.utils.data import DataLoader
from utils import *

class CNN_RV1(nn.Module):
    def __init__(self):
        super(CNN_RV1, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # in_channels, out_channels, kernel_size
        self.batch_norm1 = torch.nn.BatchNorm2d(6) # num of input_channels
        self.pool1 = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.batch_norm2 = torch.nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.batch_norm3 = torch.nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 3 * 7, 120)
        self.batch_norm4 = torch.nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        # input shape: (batch_size, channels, height, width) : (b, 3, 100, 177)
        x = self.pool1(self.batch_norm1(F.relu(self.conv1(x)))) # out: (N, 64, 6, 24, 43)
        x = self.pool2(self.batch_norm2(F.relu(self.conv2(x)))) # out: (N, 64, 16, 10, 19)
        x = self.pool3(self.batch_norm3(F.relu(self.conv3(x)))) # out: (N, 64, 32, 3, 7)
        x = x.view(-1, 32 * 3 * 7)
        x = self.batch_norm4(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


def train(verbose_interval=1):
    for epoch in range(1, num_epochs+1):
        train_loss, train_mae = 0.0, 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_mae += torch.sum(torch.abs(labels - outputs)).item()
        if epoch % verbose_interval == 0:
            train_loss /= len(train_set)
            train_mae /= len(train_set)
            val_mae, val_loss = validation()
            print('epoch: {:} | train_loss: {:.2f} | val_loss: {:.2f} | train_MAE: {:.2f} | val_MAE: {:.2f}'
                  .format(epoch + 1, train_loss, val_loss, train_mae, val_mae))
            update_history_RV1(history, epoch, train_loss, val_loss, train_mae, val_mae)


def validation():
    with torch.no_grad():
        net.train(False)
        inputs, labels = val_set[:]
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        val_mae = torch.abs(outputs - labels)
        loss = criterion(outputs, labels)
        net.train(True)
        return torch.sum(val_mae).item() / len(val_set), loss.item() / len(val_set)


def test(path):
    with torch.no_grad():
        net.train(False)
        inputs, labels = test_set[:]
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        test_mae = torch.mean(torch.abs(outputs - labels)).item()
        print("Testing: test_mae: {:.2f}".format(test_mae))
        save_test_R(outputs, labels, path)



if torch.cuda.is_available():
    dev = "cuda:0"
    print('Running on GPU')
else:
    dev = "cpu"
    print('Running on CPU')

device = torch.device(dev)

batch_size = 128
num_epochs = 2000
learning_rate = 0.001
train_set = ScaledDatasetR("uniform_train/*", height=100)

train_val_splits = [len(train_set) - round(len(train_set)*0.2), round(len(train_set)*0.2)]  # lengths of splits to be produced
history = {"epoch": [], "train_loss": [], "validation_loss": [], "train_MAE": [], "validation_MAE": []}


train_set, val_set = torch.utils.data.random_split(train_set, train_val_splits)

train_loader = DataLoader(dataset=train_set,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2)

net = CNN_RV1()
net.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
train(verbose_interval=15)

save_history(history, "history_models/CNN_RV1.json")
save_model(num_epochs, batch_size, net, optimizer, criterion, "trained_models/CNN_RV1.pt")

test_set = ScaledDatasetR("uniform_test/*", height=100)
test("test_models/CNN_RV1.txt")
