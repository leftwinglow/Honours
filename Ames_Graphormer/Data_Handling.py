import dgl
import numpy as np
from typing import Literal

def graphormer_df_creator(format: Literal["graphormer_format", "hf_format", "test_dataset"]):
    import platform
    from Data_Cleaning import to_graphormer_format
    from datasets import load_from_disk

    if platform.system() == "Linux":
        linux = True

    if format == "graphormer_format":
        if linux is True:
            dataset_processed = load_from_disk("/mnt/c/users/luke/documents/university/5th year/honours python/transformed_data/Hansen_Graphormer_DF")
        if linux is False:
            dataset_processed = load_from_disk("C:/Users/Luke/Documents/University/5th Year/Honours Python/Transformed_Data/Hansen_Graphormer_DF")

    if format == "hf_format":
        if linux is True:
            dataset_processed = to_graphormer_format("/mnt/c/users/luke/documents/university/5th year/honours python/transformed_data/Hansen_HF_Graph.pkl")
        if linux is False:
            dataset_processed = to_graphormer_format("C:/Users/Luke/Documents/University/5th Year/Honours Python/Transformed_Data/Hansen_HF_Graph.pkl")

    if format == "test_dataset":
        from datasets import load_dataset
        from transformers.models.graphormer.collating_graphormer import preprocess_item

        dataset_processed = load_dataset("OGB/ogbg-molhiv")
        dataset_processed = dataset_processed.map(preprocess_item, batched=False)

    if format != "test_dataset":
        # dataset_processed = dataset_processed.class_encode_column("y")
        dataset_processed = dataset_processed.train_test_split(test_size=0.2, seed=42)

    return dataset_processed


def get_edge_index(graph: dgl.DGLHeteroGraph) -> list[list[int]]:
    src, dst = graph.edges()
    edge_list = list(zip(src.tolist(), dst.tolist()))
    edge_index = [[edge[0] for edge in edge_list], [edge[1] for edge in edge_list]]

    return edge_index


def list_to_int32(lst: list[int | float]) -> list[np.int32]:
    return list(map(np.int32, lst))
