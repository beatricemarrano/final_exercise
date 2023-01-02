import torch
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

def mnist():
    # exchange with the corrupted mnist dataset
    train= torch.Tensor(np.load('/Users/mac/Documents/GitHub/dtu_mlops/data/corruptmnist/train_0.npz'))
    test= torch.Tensor(np.load('/Users/mac/Documents/GitHub/dtu_mlops/data/corruptmnist/test.npz'))
    
    #train = torch.randn(50000, 784)
    #test = torch.randn(10000, 784) 
    return train, test




#train= np.load('/Users/mac/Documents/GitHub/dtu_mlops/data/corruptmnist/train_0.npz')
#labels= train['labels']
#images= train['images']