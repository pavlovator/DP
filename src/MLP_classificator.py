import torch.nn as nn
import pandas as pd
from dataset_creators import *
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from utils import *
from sklearn import preprocessing

class MLP_Classificator(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out):
        super(MLP_Classificator, self).__init__()
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.dim_out = dim_out

        self.W_hid = nn.Linear(dim_in, dim_hid)
        self.W_out = nn.Linear(dim_hid, dim_out)

        self.f_hid = nn.Tanh()
        self.f_out = nn.Softmax(dim=1)

    def forward(self, x):
        a = self.W_hid(x)
        h = self.f_hid(a)
        b = self.W_out(h)
        y = self.f_out(b)
        return y

    def fit(self, n_epochs, X, T, criterion, optimizer, batch_size, validation_split, verbose = True):
        treshold = int(X.shape[0] * validation_split)
        X_train, T_train, X_val, T_val = torch.from_numpy(X[:treshold]).float(), torch.from_numpy(T[:treshold]).float(), torch.from_numpy(X[treshold:]).float(), torch.from_numpy(T[treshold:]).float()
        enc = preprocessing.OneHotEncoder(categories='auto')
        enc.fit(T)
        T_val = enc.transform(T_val).toarray()
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
                T_batch_hot = enc.transform(T_batch).toarray()
                loss = criterion(Y_batch, torch.from_numpy(T_batch_hot).float())
                loss.backward()
                optimizer.step()
                E += loss.item()
            e_train_loss.append(E/len(X_train))

            self.train(False)
            Y_val = self(X_val)
            E = criterion(Y_val, torch.from_numpy(T_val).float()).item() / len(X_val)
            e_val_loss.append(E)

            Y_train = self(X_train)
            train_acc = np.sum(Y_train.data.numpy().argmax(axis=1).reshape(-1,1) == T_train.data.numpy())/len(T_train)
            val_acc = np.sum(Y_val.data.numpy().argmax(axis=1).reshape(-1,1) == T_val.argmax(axis=1))/len(T_val)

            e_train_acc.append(train_acc)
            e_val_acc.append(val_acc)
            if verbose:
                print("epochs: {:} || train_loss: {:.5f} val_loss: {:.5f} | train_acc: {:.5f} val_acc: {:.5f}".format(ep, e_train_loss[-1], e_val_loss[-1], e_train_acc[-1], e_val_acc[-1]))
        return e_train_loss, e_val_loss, e_train_acc, e_val_acc


'''
train_set = pd.read_csv('datasets/processed/train_set_3m_processed.csv')
train_set = extract_direction(train_set, '315')

X_train, _ = get_XT(train_set, '315')
T_train = get_three_classes(train_set, '315', [5000, 15000]).reshape(-1,1)
X_train = normalize(X_train)

mlp = MLP_Classificator(19,22,3)

criterion = F.binary_cross_entropy
optimizer = optim.Adam(mlp.parameters(), lr=0.02)#, momentum=0.5)
e_train_loss, e_val_loss, e_train_acc, e_val_acc = mlp.fit(300, X_train, T_train, criterion, optimizer, 100, 0.8)
plot_NN_history(e_train_loss, e_val_loss, e_train_acc, e_val_acc)
'''
