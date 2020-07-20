from torch.utils.data import Dataset
import torch
import glob
from PIL import Image
from torchvision import transforms


class BaseDataset(Dataset):
    def __init__(self, folder_name):
        super().__init__()
        self.file_names = glob.glob(folder_name)
        self.X = []
        self.Y = []
        self.dates = []
        self.directions = []
        self.distances = []
        self.arrows = []
        for name in self.file_names:
            parts = name.split('_')
            direction = parts[2]
            distance = parts[4]
            date = parts[6]
            tensor_arrows = torch.as_tensor(list(map(float, list(parts[8].split(".")[0]))))
            self.distances.append(float(distance))
            self.dates.append(date)
            self.directions.append(int(direction))
            self.arrows.append(tensor_arrows)
        self.Y = torch.as_tensor(self.distances).unsqueeze(1)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def get_all(self):
        return self.X, self.Y

    def __len__(self):
        return len(self.Y)


class ScaledDatasetR(BaseDataset):
    def __init__(self, folder_name, width):
        super().__init__(folder_name)
        transformer = transforms.Compose([transforms.Resize(width),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.0,), (1.0,))])
        for name in self.file_names:
            image = Image.open(name)
            self.X.append(transformer(image))
        self.X = torch.stack(self.X)

class ScaledDatasetC(BaseDataset):
    def __init__(self, folder_name, width, direction):
        super().__init__(folder_name)
        self.transformer = transforms.Compose([transforms.Resize(width),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.0,), (1.0,))])
        self.current_direction = direction
        self.change_direction(self.current_direction)

    def change_direction(self, direction):
        self.current_direction = direction
        self.X, self.Y, self.indexes = [], [], []
        for idx, img_info in enumerate(zip(self.file_names, self.directions)):
            img_name, img_direction = img_info
            if img_direction == self.current_direction:
                image = Image.open(img_name)
                self.X.append(self.transformer(image))
                self.Y.append(self.arrows[idx])
                self.indexes.append(idx)
        self.X = torch.stack(self.X)
        self.Y = torch.stack(self.Y)


#test_set_R = ScaledDatasetR("uniform_test/*", 100)