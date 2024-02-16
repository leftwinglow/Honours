import pandas as pd
import torch
import torch.utils.data
import dgl.data
import dgllife as dl
from typing import Literal
import datamol as dm


class SMILES_Graph_Dataset(dgl.data.DGLDataset):
    def __init__(self, smiles: pd.Series, labels: pd.Series, graph_type: Literal["bigraph", "complete_graph"] = "bigraph", get_coords: bool | Literal["Datamol", "DGL"] = False) -> None:
        self.mol: pd.Series = smiles.apply(dm.to_mol)
        self.labels = labels

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        match graph_type:
            case "bigraph":
                self.mol = self.mol.apply(DGL_Graph_Maker().dgl_bigraph)
            case "complete_graph":
                self.mol = self.mol.apply(DGL_Graph_Maker().dgl_complete_graph)

        match get_coords:
            case False:
                pass
            case "Datamol":
                self.get_coords()
            case "DGL":
                pass

    def __getitem__(self, idx):
        graph = self.mol.iloc[idx]
        label = torch.tensor(self.labels.iloc[idx])

        return graph, label

    def __len__(self):
        return len(self.labels)

    def get_coords(self):
        from Modules.Processing_Operations import get_coordinates

        self.coords = self.mol.apply(get_coordinates)


class DGL_Graph_Maker:
    def __init__(self, node_featurizer=dl.utils.CanonicalAtomFeaturizer(), edge_featurizer=dl.utils.CanonicalBondFeaturizer(), add_self_loop=True) -> None:
        self.node_featurizer = node_featurizer
        self.edge_featurizer = edge_featurizer
        self.add_self_loop = add_self_loop
        pass

    def dgl_bigraph(self):
        lambda x: dl.utils.mol_to_bigraph(x, node_featurizer=self.node_featurizer, edge_featurizer=self.edge_featurizer, add_self_loop=self.add_self_loop)

    def dgl_complete_graph(self):
        lambda x: dl.utils.mol_to_complete_graph(x, node_featurizer=self.node_featurizer, edge_featurizer=self.edge_featurizer, add_self_loop=self.add_self_loop)


def batched_rand_sample_dataloader(dataset, train_size: float = 0.8, batch_size: int = 32):
    num_examples: float = len(dataset)
    num_train: float = int(num_examples * train_size)

    train_sampler = torch.utils.data.SubsetRandomSampler(range(num_train))
    test_sampler = torch.utils.data.SubsetRandomSampler(range(num_train, num_examples))

    train_dataloader = dgl.dataloading.GraphDataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=False)
    test_dataloader = dgl.dataloading.GraphDataLoader(dataset, batch_size=batch_size, sampler=test_sampler, drop_last=False)

    return train_dataloader, test_dataloader


class metric_plots:
    def __init__(self, DP: int = 3) -> None:
        self.DP = DP

    def sns_train_test_loss(self, train_losses, test_losses):
        import matplotlib.pyplot as plt

        plt.plot(train_losses, label="Train Loss")
        plt.plot(test_losses, label="Test Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.ylim(0, 1)

        plt.legend()
        plt.show()

    def sns_train_test_acc(self, test_accs):
        import matplotlib.pyplot as plt

        plt.plot(test_accs, label="Test Accuracy")

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        plt.ylim(0, 1)

        plt.legend()
        plt.show()
