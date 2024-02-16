import dgl
import numpy as np


def get_edge_index(graph: dgl.DGLHeteroGraph) -> list[list[int]]:
    src, dst = graph.edges()
    edge_list = list(zip(src.tolist(), dst.tolist()))
    edge_index = [[edge[0] for edge in edge_list], [edge[1] for edge in edge_list]]

    return edge_index


def list_to_int32(lst: list[int | float]) -> list[np.int32]:
    return list(map(np.int32, lst))
