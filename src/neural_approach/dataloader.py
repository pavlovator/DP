from torch.utils.data import Dataset
import torch
import glob
from PIL import Image
from torchvision import transforms


class BaseDataset(Dataset):
    def __init__(self, folder_name):
        self.file_names = glob.glob(folder_name)
        self.X = []
        self.Y = []
        self.dates = []
        self.directions = []
        for name in self.file_names:
            parts = name.split('_')
            direction = parts[2]
            distance = parts[4]
            date = parts[6].split('.')[0]
            self.Y.append(float(distance))
            self.dates.append(date)
            self.directions.append(direction)
        self._len = len(self.Y)
        self.Y = torch.as_tensor(self.Y).unsqueeze(1)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self._len



class ScaledDataset(BaseDataset):
    def __init__(self, folder_name, width=500):
        super().__init__(folder_name)
        transformer = transforms.Compose([transforms.Resize(width),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.0, ), (1, ))])
        for name in self.file_names:
            image = Image.open(name)
            self.X.append(transformer(image))
        self.X = torch.stack(self.X)

