import torch
import torchvision
import numpy as np
import os
import pickle

class DatasetEMNIST(torch.utils.data.Dataset):
    def __init__(self, is_train, len):
        super().__init__() # 62 classes
        self.data = torchvision.datasets.EMNIST(
            root='./data',
            split='byclass',
            train=is_train,
            download=True
        )
        self.labels = self.data.classes
        self.len = len

    def __len__(self):
        if self.len:
            return self.len
        return len(self.data)

    def __getitem__(self, idx):
        # list tuple np.array torch.FloatTensor
        pil_x, y_idx = self.data[idx]
        np_x = np.transpose(np.array(pil_x)).astype(np.float32)
        np_x = np.expand_dims(np_x, axis=0) / 255.0 # (1, W, H) => (1, 28, 28)
        return np_x, y_idx

class DatasetAriel(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        path_dataset = '../data/ariel_dataset.pkl'
        if not os.path.exists(path_dataset):
            os.makedirs('../data', exist_ok=True)
            download_url_to_file(
                'http://share.yellowrobot.xyz/1636095494-dml-course-2021-q4/ariel_dataset.pkl',
                path_dataset,
                progress=True
            )
        with open(path_dataset, 'rb') as fp:
            self.X, self.Y, self.labels = pickle.load(fp)
        self.labels = list(self.labels)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        x = np.expand_dims(np.transpose(self.X[idx]), axis=0).astype(np.float32)
        y = self.Y[idx]
        return x, y