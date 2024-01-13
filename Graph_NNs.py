import pandas as pd
import torch

from GNN_Utils.My_DGL_Utilities import SMILES_Graph_Dataset, batched_rand_sample_dataloader
from GNN_Utils.GNN_Models import GCN
from GNN_Utils.DGL_Training import DGL_Train_Test
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU

rega_smiles_train = pd.read_csv("Transformed_Data/rega_train.csv")


dataset = SMILES_Graph_Dataset(rega_smiles_train["smiles"], rega_smiles_train["label"])

print(next(iter(dataset)))
# train_dataloader, test_dataloader = batched_rand_sample_dataloader(dataset)

# print(next(iter(dataset))[0].ndata["h"].size(1))

# model = GCN((next(iter(dataset))[0].ndata["h"].size(1)), 16, 2).to(device)

# loss = DGL_Train_Test(model).train_step(dataset)

# DGL_Train_Test(model).train_model_crossval(dataset, 50)

# print(loss)

