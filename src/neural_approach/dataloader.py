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
            tensor_arrows = torch.as_tensor(list(map(int, list(parts[8].split(".")[0]))))
            self.distances.append(float(distance))
            self.dates.append(date)
            self.directions.append(int(direction))
            self.arrows.append(tensor_arrows)
        self._len = len(self.distances)
        self.Y = torch.as_tensor(self.distances).unsqueeze(1)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self._len


class ScaledDataset(BaseDataset):
    def __init__(self, folder_name, width):
        super().__init__(folder_name)
        transformer = transforms.Compose([transforms.Resize(width),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.0,), (1.0,))])
        for name in self.file_names:
            image = Image.open(name)
            self.X.append(transformer(image))
        self.X = torch.stack(self.X)


class ScaledClassifierDataset(ScaledDataset):
    def __init__(self, folder_name, width, direction):
        super().__init__(folder_name, width)
        self.current_direction = direction
        self.X_by_directions = {0: [], 45: [], 90: [], 135: [], 180: [], 225: [], 270: [], 315: []}
        self.Y_by_directions = {0: [], 45: [], 90: [], 135: [], 180: [], 225: [], 270: [], 315: []}
        self.dates_by_direction = {0: [], 45: [], 90: [], 135: [], 180: [], 225: [], 270: [], 315: []}
        self.distances_by_direction = {0: [], 45: [], 90: [], 135: [], 180: [], 225: [], 270: [], 315: []}
        self.arrows_by_direction = {0: [], 45: [], 90: [], 135: [], 180: [], 225: [], 270: [], 315: []}
        for i in range(self._len):
            self.X_by_directions[self.directions[i]].append(self.X[i])
            self.Y_by_directions[self.directions[i]].append(self.arrows[i])
            self.dates_by_direction[self.directions[i]].append(self.dates[i])
            self.distances_by_direction[self.directions[i]].append(self.distances[i])
            self.arrows_by_direction[self.directions[i]].append(self.arrows[i])
        self._len = len(self.X_by_directions[self.directions[i]])
        for d in range(0, 360, 45):
            self.X_by_directions[d] = torch.stack(self.X_by_directions[d])
            self.Y_by_directions[d] = torch.stack(self.Y_by_directions[d])

    def __getitem__(self, index):
        return self.X_by_directions[self.current_direction][index], self.Y_by_directions[self.current_direction][index]

    def __len__(self):
        return self._len

    def change_direction(self, direction):
        self.current_direction = direction
        self._len = len(self.X_by_directions[self.current_direction])


test_set = ScaledClassifierDataset("uniform_test/*", 100, 0)
