import os
import glob
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class MyDataLoader(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data_paths = []  
        self.labels = []      


        folders = sorted(os.listdir(data_dir))  
        self.folder_to_label = {folder: idx for idx, folder in enumerate(folders)} 

        for folder in folders:
            folder_path = os.path.join(data_dir, folder)
            if os.path.isdir(folder_path):  

                images = glob.glob(os.path.join(folder_path, '*.png')) 
                self.data_paths.extend(images)
                self.labels.extend([self.folder_to_label[folder]] * len(images))  

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img_path = self.data_paths[idx]
        label = self.labels[idx]  
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label