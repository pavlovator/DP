import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

def plot_directions(data):
    for i, direction in enumerate([0, 45, 90, 135, 180, 225, 270, 315]):
        plt.subplot(4, 2, i + 1)
        frq = data[str(direction)].value_counts()
        vals = frq.values
        plt.bar(range(len(frq.keys())), vals)
        plt.xticks(range(len(frq.keys())), list(map(int, frq.keys())), fontsize=2.9)
        plt.yticks(fontsize=3)
        plt.title(str(direction), fontsize=5)
    plt.tight_layout()
    plt.show()


def tabular_frequency(data, direction):
    return pd.DataFrame(data[str(direction)].value_counts().sort_index())


def rgbHist(imgs_orig):
    for i, vis in enumerate(imgs_orig):
        plt.subplot(2, 2, i + 1)
        plt.title(str(vis) + 'm', fontsize=6)
        img_arr = cv2.cvtColor(imgs_orig[vis], cv2.COLOR_BGR2RGB)
        histogram_r, bin_edges_r = np.histogram(img_arr[:, :, 0], bins=256, range=(0, 256))
        histogram_g, bin_edges_g = np.histogram(img_arr[:, :, 1], bins=256, range=(0, 256))
        histogram_b, bin_edges_b = np.histogram(img_arr[:, :, 2], bins=256, range=(0, 256))
        plt.plot(bin_edges_r[0:-1], histogram_r, color='r', linewidth=0.5)
        plt.plot(bin_edges_g[0:-1], histogram_g, color='g', linewidth=0.5)
        plt.plot(bin_edges_b[0:-1], histogram_b, color='b', linewidth=0.5)
        plt.xticks(fontsize=3)
        plt.yticks(fontsize=3)
    plt.show()


def greyHist(imgs_orig):
    for i, vis in enumerate(imgs_orig):
        plt.subplot(2, 2, i + 1)
        plt.title(str(vis) + 'm', fontsize=6)
        img_arr = cv2.cvtColor(imgs_orig[vis], cv2.COLOR_BGR2GRAY)
        histogram, bin_edges = np.histogram(img_arr, bins=256, range=(0, 256))
        plt.plot(bin_edges[0:-1], histogram, color='black', linewidth=0.25)
        plt.xticks(fontsize=3)
        plt.yticks(fontsize=3)
    plt.show()


def hsvHist(imgs_orig):
    for i, vis in enumerate(imgs_orig):
        plt.subplot(2, 2, i + 1)
        plt.title(str(vis) + 'm', fontsize=6)
        img_arr = cv2.cvtColor(imgs_orig[vis], cv2.COLOR_BGR2HSV)
        histogram_h, bin_edges_h = np.histogram(img_arr[:, :, 0], bins=256, range=(0, 256))
        histogram_s, bin_edges_s = np.histogram(img_arr[:, :, 1], bins=256, range=(0, 256))
        histogram_v, bin_edges_v = np.histogram(img_arr[:, :, 2], bins=256, range=(0, 256))
        plt.plot(bin_edges_h[0:-1], histogram_h, color='b', linewidth=0.5)
        plt.plot(bin_edges_s[0:-1], histogram_s, color='black', linewidth=0.5)
        plt.plot(bin_edges_v[0:-1], histogram_v, color='orange', linewidth=0.5)
        plt.legend(('hue', 'saturation', 'brightness'), prop={'size': 6})
        plt.xticks(fontsize=3)
        plt.yticks(fontsize=3)
    plt.show()


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def variance_of_sobelx(image):
    return cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5).var()

def variance_of_sobely(image):
    return cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5).var()

def get_random_img_sample(data, direction, visibility):
    rows = data[data[direction] == visibility][[direction, 'date']]
    return rows.sample().date.values[0]

def get_image(str_date, direction):
    datetime = pd.to_datetime(str_date)
    img_name = "panasonic_fullhd_{:}_{:}.jpg".format(datetime.strftime('%Y%m%d%H%M'), direction.zfill(3))
    return img_name


def create_grey_space(data, direction):
    hists = []
    grey_bins = np.array(list(range(257)))
    for date in data:
        img_name = get_image(date, direction)
        img = cv2.imread('pics/{:}'.format(img_name))
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_vectorized = img_grey.ravel()
        h, b = np.histogram(img_vectorized, bins=grey_bins)
        hists.append(h)
    return np.array(hists, dtype=np.float32)

def create_hsv_space(data, direction, dim):
    '''
    For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]
    '''
    hists = []
    for date in data:
        img_name = get_image(date, direction)
        img = cv2.imread('pics/{:}'.format(img_name))
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if dim == 'h':
            img_hsv = img_hsv[:, :, 0]
            hsv_bins = np.array(list(range(181)))
        elif dim == 's':
            img_hsv = img_hsv[:, :, 1]
            hsv_bins = np.array(list(range(257)))
        elif dim == 'v':
            img_hsv = img_hsv[:, :, 2]
            hsv_bins = np.array(list(range(257)))
        img_vectorized = img_hsv.ravel()
        h, b = np.histogram(img_vectorized, bins=hsv_bins)
        hists.append(h)
    return np.array(hists, dtype=np.float32)

def normalize(data_array):
    data_array -= data_array.mean(axis=0)
    data_array /= data_array.std(axis=0)
    return data_array

def pca_dims(data_array, comps):
    pca = PCA(n_components=comps)
    X_tranformed = pca.fit_transform(data_array)
    return X_tranformed

def plot_PCs_matrix(pca_data, targets):
    k = pca_data.shape[1]
    fig, axs = plt.subplots(k-1, k-1)
    for i in range(k-1):
        for j in range(k-1):
            if i <= j:
                ax = axs[i, j].scatter(pca_data[:, i], pca_data[:, j+1], c=targets, s=1, cmap='viridis')
                axs[i, j].set_xlabel("PC{:}".format(i+1), fontsize=6)
                axs[i, j].set_ylabel("PC{:}".format(j+2), fontsize=6)
                axs[i, j].tick_params(axis='both', which='major', labelsize=6)
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
            else:
                axs[i, j].plot()
                axs[i, j].axis('off')
    cbar_ax = fig.add_axes([0.05, 0.08, 1-(1/(k-1)+0.05), 0.02])
    fig.colorbar(ax, ax=axs.ravel().tolist(), cax=cbar_ax, orientation="horizontal")
    plt.show()

def plot_NN_history(e_train_loss, e_val_loss, e_train_acc, e_val_acc):
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(e_train_loss)
    axs[0].plot(e_val_loss)
    axs[0].set_xlabel("epochs", fontsize=8)
    axs[0].set_ylabel("MSE loss", fontsize=8)
    axs[0].legend(("train loss", "validation loss"))
    axs[1].plot(e_train_acc)
    axs[1].plot(e_val_acc)
    axs[1].set_xlabel("epochs", fontsize=8)
    axs[1].set_ylabel("MAE accuracy", fontsize=8)
    axs[1].legend(("train acc", "validation acc"))

    plt.show()

def get_three_classes(data, direction, bins):
    data[direction + 'class'] = np.nan
    data.loc[data[direction] < bins[0], direction + 'class'] = 0
    data.loc[np.logical_and(bins[0] <= data[direction], bins[1] >= data[direction]), direction + 'class'] = 1
    data.loc[data[direction] > bins[1], direction + 'class'] = 2
    values = data[direction+'class'].values
    data.drop(columns=[direction+'class'], inplace=True)
    return values


def get_XT(data, direction):
    return data.loc[:, data.columns != direction].values, data.loc[:, direction].values

def plot_real_pred(T, Y):
    plt.scatter(T,Y)
    plt.xlabel('target')
    plt.ylabel('prediction')
    plt.show()

'''
train_set = pd.read_csv('datasets/processed/train_set_3m_processed.csv')
test_set = pd.read_csv('datasets/processed/test_set_3m_processed.csv')
'''