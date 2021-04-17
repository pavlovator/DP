from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
from utils import *
from dataset_creators import *
from matplotlib import pyplot as plt

def plot_hyperparameters_regressor(X, T, validation=0.8):
    treshold = int(X.shape[0]*validation)
    X = normalize(X)
    X_train, T_train, X_val, T_val = X[:treshold], T[:treshold], X[treshold:], T[treshold:]
    distances = ['euclidean', 'manhattan', 'chebyshev']
    train_err = {'euclidean':[], 'manhattan':[], 'chebyshev':[]}
    val_err = {'euclidean':[], 'manhattan':[], 'chebyshev':[]}
    x = list(range(2, 22))
    for m in distances:
        for i in x:
            model = KNeighborsRegressor(n_neighbors=i, metric=m)
            model.fit(X_train, T_train)
            Y_train = model.predict(X_train)
            Y_val = model.predict(X_val)
            mae_val = mean_absolute_error(Y_val, T_val)
            mae_train = mean_absolute_error(Y_train, T_train)
            train_err[m].append(mae_train)
            val_err[m].append(mae_val)
        plt.plot(x, train_err[m], label=m+'_train')
        plt.plot(x, val_err[m], '--', label=m+'_val')
        plt.xlabel("k")
        plt.xticks(x)
        plt.ylabel("MAE")
    plt.legend(loc="lower right")
    plt.show()





'''
train_set = pd.read_csv('datasets/processed/train_set_3m_processed.csv')
train_90 = extract_direction(train_set, '315')
X_train, T_train = get_XT(train_90, '315')
plot_hyperparameters_regressor(X_train, T_train)
'''