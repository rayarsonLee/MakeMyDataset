import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from dataset import *
from net import *

# 读取数据集
# dataset = pd.read_csv("dataset_sequence.csv")
# features = dataset.iloc[:, :20]
# labels = dataset.iloc[:, 20]
# features = features.values
# labels = labels.values


model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

make_train_loader()
