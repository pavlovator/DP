from torch.utils.data import Dataset
from utils.xml import XML
from utils.configs import OutputConfig
import torch
import glob
from PIL import Image
from torchvision import transforms
import numpy as np
import resource


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
                    x = Image.open(img_folder + "dir_{:}_date_{:}.jpg".format(dir, date))
                    self.raw_Y.append(y)
                    self.X.append(self.transformer(x))
                    self.directions.append(dir)
                    self.dates.append(int(date))
                    self.indexes.append(i)
                    i += 1
                    #print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, i)
            else:
                self.bad_dates.append(xml.get_date())
        self.indexes = torch.tensor(self.indexes)
        self.X = torch.stack(self.X)
        self.directions = torch.tensor(self.directions)
        self.dates = torch.tensor(self.dates)


class UnifiedDataset(Dataset, BaseDataset):
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
        return self.indexes[index], self.X[index], self.Y[index]

    def __len__(self):
        return len(self.Y)

    def get_date_direction(self, indexes):
        '''
        :param indexes: list of indexis mapping one-to-one idx -> date, diection
        :return: dates and directions corresponding to indices. (list dates, list directions)
        '''
        return self.dates[indexes].tolist(), self.directions[indexes].tolist()


'''
out_conf = OutputConfig("files/output_config.json")
xml_folder =  "../data/_testing_xmls/*"
img_folder = "../data/pics/"
bd = UnifiedDataset(xml_folder, 100, img_folder, out_conf)
'''
