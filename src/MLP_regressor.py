import torch.nn as nn
import pandas as pd
from dataset_creators import *
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from utils import *

class MLP_Regressor(nn.Module):

    def __init__(self, dim_in, dim_hid, dim_out):
        super(MLP_Regressor, self).__init__()
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.dim_out = dim_out

        self.W_hid = nn.Linear(dim_in, dim_hid)
        self.W_out = nn.Linear(dim_hid, dim_out)
        self.d1 = nn.Dropout(0.2)
        self.f_hid = nn.Tanh()
        self.f_out = nn.ReLU()

    def forward(self, x):
        x = self.W_hid(x)
        x = self.f_hid(x)
        x = self.d1(x)
        x = self.W_out(x)
        x = self.f_out(x)
        return x

    def fit(self, n_epochs, X, T, criterion, optimizer, batch_size, validation_split, verbose=True):
        treshold = int(X.shape[0] * validation_split)
        X_train, T_train, X_val, T_val = torch.from_numpy(X[:treshold]).float(), torch.from_numpy(T[:treshold]).float(), torch.from_numpy(X[treshold:]).float(), torch.from_numpy(T[treshold:]).float()
        self.train()
        n_batches = int(np.ceil(X_train.shape[0] / batch_size))
        e_train_loss, e_val_loss = [], []
        e_train_acc, e_val_acc = [], []
        for ep in range(n_epochs):
            rand_idx = np.random.permutation(X_train.shape[0])
            X_train = X_train[rand_idx]
            T_train = T_train[rand_idx]
            E = 0
            for i in range(n_batches):
                X_batch, T_batch = X_train[i*batch_size: (i+1)*batch_size], T_train[i*batch_size: (i+1)*batch_size]
                optimizer.zero_grad()
                Y_batch = self(X_batch)
                loss = criterion(Y_batch, T_batch.reshape(T_batch.shape[0],1))
                loss.backward()
                optimizer.step()
                E += loss.item()
            e_train_loss.append(E/len(X_train))

            self.train(False)
            Y_train = self(X_train)
            e_train_acc.append(np.mean(np.abs(Y_train.data.numpy().flatten()-T_train.data.numpy().flatten())))

            if validation_split == 1:
                if ep % 100 == 0 and verbose:
                    print("epochs: {:} || train_loss: {:.2f} | train_acc: {:.2f}".format(ep, e_train_loss[-1], e_train_acc[-1]))
            else:
                Y_val = self(X_val)
                E = criterion(Y_val, T_val.reshape(T_val.shape[0], 1)).item() / len(X_val)
                e_val_loss.append(E)
                e_val_acc.append(np.mean(np.abs(Y_val.data.numpy().flatten() - T_val.data.numpy().flatten())))
                if ep % 100 == 0 and verbose:
                    print("epochs: {:} || train_loss: {:.2f} val_loss: {:.2f} | train_acc: {:.2f} val_acc: {:.2f}".format(ep, e_train_loss[-1], e_val_loss[-1], e_train_acc[-1], e_val_acc[-1]))

        return e_train_loss, e_val_loss, e_train_acc, e_val_acc







'''
train_set = pd.read_csv('datasets/processed/train_set_3m_processed.csv')
test_set = pd.read_csv('datasets/processed/test_set_3m_processed.csv')
train_set = extract_direction(train_set, '315')
test_set = extract_direction(test_set, '315')
X_train, T_train = get_XT(train_set, '315')
X_test, T_test = get_XT(test_set, '315')

X_test = normalize(X_test)
X_train = normalize(X_train)

#--
mlp = MLP_Regressor(19,125,1)

criterion = F.mse_loss
optimizer = optim.Adam(mlp.parameters(), lr=0.01)
e_train_loss, e_val_loss, e_train_acc, e_val_acc = mlp.fit(5500, X_train, T_train, criterion, optimizer, 128, 0.8)
plot_NN_history(e_train_loss, e_val_loss, e_train_acc, e_val_acc)


Y_test = mlp(torch.from_numpy(X_test).float())
Y_test = Y_test.data.numpy().flatten()
print("Priemerna absolutna chyba (MAE) na testovacej mnozine: {:.2f}".format(np.mean(np.abs(Y_test-T_test))))

#----
mlp = MLP_Regressor(19,125,1)

criterion = F.mse_loss
optimizer = optim.Adam(mlp.parameters(), lr=0.01)
e_train_loss, e_val_loss, e_train_acc, e_val_acc = mlp.fit(5500, X_train, T_train, criterion, optimizer, 128, 1)
Y_test = mlp(torch.from_numpy(X_test).float())
Y_test = Y_test.data.numpy().flatten()
print("Priemerna absolutna chyba (MAE) na testovacej mnozine: {:.2f}".format(np.mean(np.abs(Y_test-T_test))))
'''
