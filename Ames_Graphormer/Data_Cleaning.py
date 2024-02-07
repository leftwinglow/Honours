import pandas as pd
import dgllife as dl
import datamol as dm
import numpy as np
from datasets import Dataset, load_dataset

import os


def to_transformed(raw_hansen_data: str | os.PathLike, save_csv: bool = False) -> pd.DataFrame:
    hansen_raw = pd.read_csv(raw_hansen_data)

    hansen_raw = hansen_raw.drop(hansen_raw.columns[1], axis=1)
    hansen_raw.columns = ["smiles", "ames"]
    hansen_raw.smiles = hansen_raw.smiles.apply(dm.to_mol)
    hansen_transformed = hansen_raw

    if save_csv is True:
        hansen_transformed.to_csv("C:/Users/Luke/Documents/University/5th Year/Honours Python/Transformed_Data/Hansen_Ames.csv", index=False)

    return hansen_transformed


def hansen_to_graph(hansen_transformed: pd.DataFrame | os.PathLike) -> pd.DataFrame:
    if isinstance(hansen_transformed, os.PathLike):
        hansen_dgl_graph = pd.read_csv(hansen_transformed)
    elif isinstance(hansen_transformed, pd.DataFrame):
        hansen_dgl_graph = hansen_transformed
    else:
        raise ValueError("transformed_hansen_data must be a pandas DataFrame or a Path object")

    hansen_dgl_graph.smiles = hansen_dgl_graph.smiles.apply(lambda x: dl.utils.mol_to_bigraph(x, node_featurizer=dl.utils.CanonicalAtomFeaturizer(), edge_featurizer=dl.utils.CanonicalBondFeaturizer()))

    return hansen_dgl_graph


def convert_to_int32(lst):
    return list(map(np.int32, lst))


def to_hf(hansen_graph: pd.DataFrame, save_pickle: bool = True, save_csv: bool = False) -> pd.DataFrame:
    from Data_Handling import get_edge_index

    hansen_hf_graph = pd.DataFrame(columns=["edge_index", "node_feat", "edge_attr", "y", "num_nodes"])

    hansen_hf_graph = hansen_graph.assign(
        edge_index=hansen_graph.smiles.apply(get_edge_index),
        node_feat=hansen_graph.smiles.apply(lambda x: x.ndata["h"].tolist()),
        edge_attr=hansen_graph.smiles.apply(lambda x: x.edata["e"].tolist()),
        y=hansen_graph.ames.astype(np.int32),
        num_nodes=hansen_graph.smiles.apply(lambda x: x.number_of_nodes()),
    )

    hansen_hf_graph.edge_index = hansen_hf_graph.edge_index.apply(convert_to_int32)
    hansen_hf_graph.node_feat = hansen_hf_graph.node_feat.apply(convert_to_int32)
    hansen_hf_graph.edge_attr = hansen_hf_graph.edge_attr.apply(convert_to_int32)

    hansen_hf_graph.num_nodes = hansen_hf_graph.num_nodes.astype(np.int32)

    hansen_hf_graph = hansen_hf_graph.drop(["smiles", "ames"], axis=1)

    if save_csv is True:
        hansen_hf_graph.to_csv("C:/Users/Luke/Documents/University/5th Year/Honours Python/Transformed_Data/Hansen_HF_Graph.csv", index=False)
    if save_pickle is True:
        hansen_hf_graph.to_pickle("C:/Users/Luke/Documents/University/5th Year/Honours Python/Transformed_Data/Hansen_HF_Graph.pkl")

    return hansen_hf_graph


def to_graphormer_format(hf_graph_dataset: pd.DataFrame | str | os.PathLike) -> Dataset:
    from transformers.models.graphormer.collating_graphormer import preprocess_item

    if isinstance(hf_graph_dataset, str | os.PathLike):
        hf_graph_dataset = pd.read_pickle(hf_graph_dataset)
    else:
        pass

    dataset = Dataset.from_pandas(hf_graph_dataset)
    # dataset = dataset.map(preprocess_item)

    return dataset


hansen_transformed = to_transformed("C:/Users/Luke/Documents/University/5th Year/Honours Python/Raw_Data/Hansen_Ames.csv", True)
hansen_graph = hansen_to_graph(hansen_transformed)
hansen_hf_format = to_hf(hansen_graph)
hansen_graphormer_format = to_graphormer_format(hansen_hf_format)

print(hansen_hf_format.head())


