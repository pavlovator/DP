from sklearn.neighbors import KNeighborsClassifier
from utils import *
from dataset_creators import *

def plot_hyperparameters_classifier(X, T, validation=0.8):
    treshold = int(X.shape[0]*validation)
    X = normalize(X)
    X_train, T_train, X_val, T_val = X[:treshold], T[:treshold], X[treshold:], T[treshold:]
    distances = ['euclidean', 'manhattan', 'chebyshev']
    train_err = {'euclidean':[], 'manhattan':[], 'chebyshev':[]}
    val_err = {'euclidean':[], 'manhattan':[], 'chebyshev':[]}
    x = list(range(3, 22,2))
    for m in distances:
        for i in x:
            model = KNeighborsClassifier(n_neighbors=i, metric=m)
            model.fit(X_train, T_train)
            Y_train = model.predict(X_train)
            Y_val = model.predict(X_val)
            train_acc = np.mean((Y_train == T_train))*100
            val_acc = np.mean((Y_val == T_val))*100
            train_err[m].append(train_acc)
            val_err[m].append(val_acc)
        plt.plot(x, train_err[m], label=m+'_train')
        plt.plot(x, val_err[m], '--', label=m+'_val')
        plt.xlabel("k")
        plt.xticks(x)
        plt.ylabel("%")
    plt.legend(loc="upper right")
    plt.show()



train_set = pd.read_csv('datasets/processed/train_set_3m_processed.csv')
train_set = extract_direction(train_set, '315')
classes = get_three_classes(train_set, '315', [5000, 15000])
train_set['315'] = classes
X_train, T_train = get_XT(train_set, '315')

plot_hyperparameters_classifier(X_train, T_train)
