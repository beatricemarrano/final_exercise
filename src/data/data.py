import torch
#import wget
import numpy as np
import os

from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class CorruptMnist(Dataset):
    #def __init__(self, train):
        #path= '/Users/mac/Documents/GitHub/final_exercise/data/processed'
        #if train:
            #data = torch.load(os.path.join(path, "data_train.pkl"))
            #targets = torch.load(os.path.join(path, "targets_train.pkl"))
        #else:
            #data = torch.load(os.path.join(path, "data_test.pkl"))
            #targets = torch.load(os.path.join(path, "targets_test.pkl"))
            
        #self.data = data
        #self.targets = targets
    
    def __init__(self, path: str, type: str = "train") -> None:
        #path= '/Users/mac/Documents/GitHub/final_exercise/data/processed'
        if type == "train":
            file_data= os.path.join(path, "data_train.pkl")
            file_targets = os.path.join(path, "targets_train.pkl")
        elif type == "test":
            file_data= os.path.join(path, "data_test.pkl")
            file_targets = os.path.join(path, "targets_test.pkl")
        else:
            raise Exception(f"Unknown Dataset type: {type}")

        self.targets = torch.load(file_targets)
        self.data = torch.load(file_data)
    
    def __len__(self):
        return self.targets.numel()
    
    def __getitem__(self, idx):
        return self.data[idx].float(), self.targets[idx]


#if __name__ == "__main__":
    #dataset_train = CorruptMnist(train=True)
    #dataset_test = CorruptMnist(train=False)
    #print(dataset_train.data.shape)
    #print(dataset_train.targets.shape)
    #print(dataset_test.data.shape)
    #print(dataset_test.targets.shape)
    
    
#Implementing Pytorch Lightning
class CorruptMnistDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int = 32):
        super().__init__()
        self.data_path = os.path.join(data_path, "processed")
        self.batch_size = batch_size
        self.cpu_cnt = os.cpu_count() or 2

    def prepare_data(self) -> None:
        if not os.path.isdir(self.data_path):
            raise Exception("data is not prepared")

    def setup(self, stage: Optional[str] = None) -> None:
        self.trainset = CorruptMnist(self.data_path, "train")
        self.testset = CorruptMnist(self.data_path, "test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.trainset, batch_size=self.batch_size, num_workers=self.cpu_cnt
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.testset, batch_size=self.batch_size, num_workers=self.cpu_cnt
        )
        