import os, csv, PIL

from torch.utils.data import Dataset


class NIPA_Dataset(Dataset):
    def __init__(self, data_type, transform=None):
        if data_type == 'train':
            self.img_dir = './train/'
            self.label_dir = './train/train.tsv'
        elif data_type == 'val':
            self.img_dir = './test/'
            self.label_dir = './test/test.tsv'
            
        label_dic = {'3_5':0, '3_20':1, '4_2':2, '4_7':3, '4_11':4, '5_8':5, '7_1':6, '7_20':7, '8_6':8, '8_9':9, '10_20':10,
                     '11_14':11, '13_1':12, '13_6':13, '13_9':14, '13_15':15, '13_16':16, '13_17':17, '13_18':18, '13_20':19}
        self.img2label = {}
        with open(self.label_dir, 'r') as tsvfile:
            for line in csv.reader(tsvfile, delimiter='\t'):
                img, label_p, label_d = line
                label = label_p + '_' + label_d
                self.img2label[img] = label_dic[label]
        self.imgs = list(self.img2label.keys())

        self.n_labels = 20
        self.transform = transform

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img = PIL.Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        label = self.img2label[self.imgs[idx]]
        return img, label

    def __len__(self):
        return len(self.imgs)