import sys

sys.path.append("C:/Users/Luke/Documents/University/5th Year/Honours Python")

import dgl
import torch
import pandas as pd
from GNN_Utils import My_DGL_Utilities

test_dataframe = pd.read_csv("Transformed_Data/rega_train.csv", nrows=10)

print(test_dataframe)


def test_smiles_graph_dataset_general() -> None:
    dataloader = My_DGL_Utilities.SMILES_Graph_Dataset(test_dataframe["smiles"], test_dataframe["label"])
    print(next(iter(dataloader)))
    assert isinstance(next(iter(dataloader))[0], dgl.DGLGraph) and isinstance(next(iter(dataloader))[1], torch.Tensor)


def test_random_sampled_dataloader() -> None:
    dataloader = My_DGL_Utilities.SMILES_Graph_Dataset(test_dataframe["smiles"], test_dataframe["label"])
    train_dataloader, test_dataloader = My_DGL_Utilities.batched_rand_sample_dataloader(dataloader)
    assert isinstance(next(iter(train_dataloader))[0], dgl.DGLGraph) and isinstance(next(iter(train_dataloader))[1], torch.Tensor)
    assert isinstance(next(iter(test_dataloader))[0], dgl.DGLGraph) and isinstance(next(iter(test_dataloader))[1], torch.Tensor)
