from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Train_148(Dataset):
    def __init__(self, infos, transform=None):
        self.train_dir = '../../data/.train/.task148/data/train/images/'
        self.train_csv = '../../data/.train/.task148/data/train/train.csv'
        
        self.infos = infos
        self.transform = transform

    def __getitem__(self, idx):
        img, meta, label = self.infos[idx]
        img = Image.open(self.train_dir + img + '.png').convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        return img, torch.Tensor(meta), label

    def __len__(self):
        return len(self.infos)
    
    
class Test_148(Dataset):
    def __init__(self, infos, transform=None):
        self.test_dir = '../../data/.train/.task148/data/test/images/'
        self.test_csv = '../../data/.train/.task148/data/test/test.csv'
        
        self.infos = infos
        self.transform = transform

    def __getitem__(self, idx):
        img, meta = self.infos[idx]
        img = Image.open(self.test_dir + img + '.png').convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        return img, torch.Tensor(meta)

    def __len__(self):
        return len(self.infos)