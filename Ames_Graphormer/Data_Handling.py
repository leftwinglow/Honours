import pandas as pd
import torch
import dgl, dgl.data
import dgllife as dl
import datamol as dm
from typing import Literal

class SMILES_Graph_Dataset(dgl.data.DGLDataset):
    def __init__(
        self, smiles: pd.Series, labels: pd.Series, graph_type: Literal['bigraph', 'complete_graph'] = 'bigraph', get_coords: bool | Literal['Datamol', 'DGL'] = False) -> None:
        self.mol: pd.Series = smiles.apply(dm.to_mol)
        self.labels = labels

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        match graph_type:
            case 'bigraph':
                self.mol = self.mol.apply(DGL_Graph_Maker().dgl_bigraph)
            case 'complete_graph':
                self.mol = self.mol.apply(DGL_Graph_Maker().dgl_complete_graph)

        match get_coords:
            case False:
                pass
            case 'Datamol':
                self.get_coords()
            case 'DGL':
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

class DGL_Graph_Maker():
    def __init__(self, node_featurizer = dl.utils.CanonicalAtomFeaturizer(), edge_featurizer = dl.utils.CanonicalBondFeaturizer(), add_self_loop = True) -> None:
        self.node_featurizer = node_featurizer
        self.edge_featurizer = edge_featurizer
        self.add_self_loop = add_self_loop
        pass

    def dgl_bigraph(self):
        lambda x: dl.utils.mol_to_bigraph(x, node_featurizer=self.node_featurizer, edge_featurizer=self.edge_featurizer, add_self_loop=self.add_self_loop)

    def dgl_complete_graph(self):
        lambda x: dl.utils.mol_to_complete_graph(x, node_featurizer=self.node_featurizer, edge_featurizer=self.edge_featurizer, add_self_loop=self.add_self_loop)

def get_edge_index(graph):
    src, dst = graph.edges()
    edge_list = list(zip(src.tolist(), dst.tolist()))
    edge_index = [[edge[0] for edge in edge_list], [edge[1] for edge in edge_list]]

    return edge_index