from utils import *
import cv2

def create_dataset_train_test(data, train_name, test_name, split = 0.75):
    data = data.dropna()
    data = data.sample(frac=1).reset_index(drop=True)
    N = data.shape[0]
    threshold = int(N * split)
    train = data[:threshold]
    test = data[threshold:]
    train.to_csv(train_name, index=False)
    test.to_csv(test_name, index=False)

def add_hour_feature(data):
    data['hour'] = pd.to_datetime(data.date).dt.hour

def add_varinaces(data):
    for direction in ['0', '45', '90', '135', '180', '225', '270', '315']:
        data['var_' + direction + '_lap'] = np.nan
        data['var_' + direction + '_sobelx'] = np.nan
        data['var_' + direction + '_sobely'] = np.nan
    for index, row in data.iterrows():
        for direction in ['0', '45', '90', '135', '180', '225', '270', '315']:
            img_name = get_image(row['date'], direction)
            img = cv2.imread('pics/{:}'.format(img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            data.loc[index, 'var_' + direction + '_lap'] = variance_of_laplacian(img)
            data.loc[index, 'var_' + direction + '_sobelx'] = variance_of_sobelx(img)
            data.loc[index, 'var_' + direction + '_sobely'] = variance_of_sobely(img)

def add_pca_HSV(data_train, data_test, dimension, components=5):
    def test_transformation():
        data_hsb_test = create_hsv_space(data_test.date, direction, dimension)
        normalized_test = normalize(data_hsb_test)
        X_tranformed_test = pca.transform(normalized_test)
        for k in range(components):
            PC_k_test = X_tranformed_test[:, k]
            col_name_test = "PC{:}_{:}_{:}".format(k + 1, direction, dimension)
            data_test[col_name_test] = PC_k_test

    for direction in ['0', '45', '90', '135', '180', '225', '270', '315']:
        data_hsv = create_hsv_space(data_train.date, direction, dimension)
        normalized = normalize(data_hsv)
        pca = PCA(n_components=components)
        X_tranformed = pca.fit_transform(normalized)
        for k in range(components):
            PC_k = X_tranformed[:, k]
            col_name = "PC{:}_{:}_{:}".format(k+1, direction, dimension)
            data_train[col_name] = PC_k
        test_transformation()


def process():
    data_train = pd.read_csv('datasets/raw/train_set_3m.csv')
    data_test = pd.read_csv('datasets/raw/test_set_3m.csv')
    add_hour_feature(data_train)
    add_hour_feature(data_test)
    add_pca_HSV(data_train, data_test, 'h')
    add_pca_HSV(data_train, data_test, 's')
    add_pca_HSV(data_train, data_test, 'v')
    add_varinaces(data_train)
    add_varinaces(data_test)
    data_train.to_csv("datasets/processed/train_set_3m_processed.csv", index=False)
    data_test.to_csv("datasets/processed/test_set_3m_processed.csv", index=False)

def extract_direction(data, direction):
    columns = data.filter(regex='.*_{:}_*.'.format(direction), axis=1).columns.to_list()
    columns += [direction, 'hour']
    return data[columns]


#train_set = pd.read_csv('datasets/processed/train_set_3m_processed.csv')
#test_set = pd.read_csv('datasets/processed/test_set_3m_processed.csv')
