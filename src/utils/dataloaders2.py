from torch.utils.data import Dataset
from utils.xml import XML
from utils.configs import OutputConfig
import torch
import glob
from PIL import Image
from torchvision import transforms


class BaseDataset:
    '''
    Base dataset class loads all XML files from folder.
    Wrongly annotated xmls (not consistent with config) are omitted.
    (direction[i], dates[i]) uniquely describes every situation.
    '''
    def __init__(self, xmls_folder, height, img_folder, config):
        self.transformer = transforms.Compose([transforms.Resize(height),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.0,), (1.0,))])

        self.xml_file_names = glob.glob(xmls_folder)
        self.config = config
        self.max_output = self.config.max_output_length()
        self.X = []
        self.raw_Y = []
        self.directions = []
        self.dates = []
        self.indexes = []
        self.bad_dates = []
        i = 0
        for file in self.xml_file_names:
            xml = XML(file)
            if config.is_consistent(xml):
                for dir in range(0, 360, 45):
                    date = xml.get_date()
                    y = xml.get_direction_output(dir, config)
                    self.raw_Y.append(y)
                    self.X.append(img_folder + "dir_{:}_date_{:}.jpg".format(dir, date)) #toto
                    self.directions.append(dir)
                    self.dates.append(int(date))
                    self.indexes.append(i)
                    i += 1
            else:
                self.bad_dates.append(xml.get_date())
        self.indexes = torch.tensor(self.indexes)
        self.directions = torch.tensor(self.directions)
        self.dates = torch.tensor(self.dates)


class VNET1Dataset(Dataset, BaseDataset):
    '''
    Dataset with unified number of outputs compatible with single NN
    '''
    def __init__(self, xmls_folder, height, img_folder, config):
        Dataset.__init__(self)
        BaseDataset.__init__(self, xmls_folder, height, img_folder, config)
        self.Y = []
        for y in self.raw_Y:
            zeros = torch.zeros(self.max_output)
            zeros[torch.arange(len(y))] = torch.tensor(y, dtype=torch.float32)
            self.Y.append(zeros)
        self.Y = torch.stack(self.Y)

    def __getitem__(self, index):
        x = Image.open(self.X[index])
        x = self.transformer(x)
        return self.indexes[index], x, self.Y[index]

    def __len__(self):
        return len(self.Y)

    def get_date_direction(self, indexes):
        '''
        :param indexes: list of indexis mapping one-to-one idx -> date, diection
        :return: dates and directions corresponding to indices. (list dates, list directions)
        '''
        return self.dates[indexes].tolist(), self.directions[indexes].tolist()

class VNET2Dataset(VNET1Dataset):
    '''
    dataset for second approach with gausslike indices
    '''
    def __init__(self, xmls_folder, height, img_folder, config, gauss_folder):
        VNET1Dataset.__init__(self, xmls_folder, height, img_folder, config)
        self.transgrey = transforms.ToTensor()
        self.gauss_folder = gauss_folder

    def __getitem__(self, index):
        gpath = self.gauss_folder + "{:}.jpg".format(self.directions[index].tolist())
        gmap = self.transgrey(Image.open(gpath))
        x = Image.open(self.X[index])
        x = self.transformer(x)
        return self.indexes[index], torch.cat((x, gmap), 0), self.Y[index]


class VNET3Dataset(VNET2Dataset):
    '''
    dataset for third approach with highpass edges
    '''
    def __init__(self, xmls_folder, height, img_folder, config, gauss_folder, highpass_folder):
        VNET2Dataset.__init__(self, xmls_folder, height, img_folder, config, gauss_folder)
        self.highpass_folder = highpass_folder

    def __getitem__(self, index):
        gpath = self.gauss_folder + "{:}.jpg".format(self.directions[index].tolist())
        gmap = self.transgrey(Image.open(gpath))
        path_hp = self.highpass_folder + self.X[index].split('/')[-1]
        hp = self.transgrey(Image.open(path_hp))
        x = Image.open(self.X[index])
        x = self.transformer(x)
        map_whole = torch.cat((x, gmap, hp), 0)
        return self.indexes[index], map_whole, self.Y[index]