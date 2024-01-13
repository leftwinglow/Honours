import sys
sys.path.append('C:/Users/Luke/Documents/University/5th Year/Honours Python')

import pandas as pd
from GNN_Utils.DGL_Training import DGL_Train_Test
from GNN_Utils.My_DGL_Utilities import SMILES_Graph_Dataset
from GNN_Utils.GNN_Models import GCN

# def test_train_step() -> None:
#     rega_smiles_train = pd.read_csv("Transformed_Data/rega_train.csv")
#     dataloader = SMILES_Graph_Dataset(rega_smiles_train["smiles"], rega_smiles_train["label"])
#     DGL_Train_Test(GCN).test_step(dataloader)

# print(test_train_step())