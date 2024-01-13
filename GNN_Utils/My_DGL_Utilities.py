import pandas as pd
import torch
import torch.utils.data
import dgl.data
import dgllife as dl
from typing import Literal
import datamol as dm


class SMILES_Graph_Dataset(dgl.data.DGLDataset):
    def __init__(
        self, smiles: pd.Series, labels: pd.Series, graph_type: Literal["bigraph", "complete_graph"] = "bigraph", get_coords: bool = False
    ) -> None:
        self.mol: pd.Series = smiles.apply(dm.to_mol)
        self.labels = labels

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        match graph_type:
            case "bigraph":
                self.mol = self.mol.apply(
                    lambda x: dl.utils.mol_to_bigraph(
                        x, node_featurizer=dl.utils.CanonicalAtomFeaturizer(), edge_featurizer=dl.utils.CanonicalBondFeaturizer()
                    )
                )
            case "complete_graph":
                self.mol = self.mol.apply(
                    lambda x: dl.utils.mol_to_complete_graph(
                        x, node_featurizer=dl.utils.CanonicalAtomFeaturizer(), edge_featurizer=dl.utils.CanonicalBondFeaturizer()
                    )
                )

        if get_coords is False:
            pass
        else:
            self.get_coords()

    def __getitem__(self, idx):
        graph = self.mol.iloc[idx]
        label = torch.tensor(self.labels.iloc[idx])

        return graph, label

    def __len__(self):
        return len(self.labels)

    def get_coords(self):
        from Modules.Processing_Operations import get_coordinates

        self.coords = self.mol.apply(get_coordinates)


def batched_rand_sample_dataloader(dataset, train_size: float = 0.8, batch_size: int = 32):
    num_examples: float = len(dataset)
    num_train: float = int(num_examples * train_size)

    train_sampler = torch.utils.data.SubsetRandomSampler(range(num_train))
    test_sampler = torch.utils.data.SubsetRandomSampler(range(num_train, num_examples))

    train_dataloader = dgl.dataloading.GraphDataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=False)
    test_dataloader = dgl.dataloading.GraphDataLoader(dataset, batch_size=batch_size, sampler=test_sampler, drop_last=False)

    return train_dataloader, test_dataloader