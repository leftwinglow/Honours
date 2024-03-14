import pandas as pd
import datamol as dm
import os
import random

transformed_dataframe = pd.read_csv("C:/Users/Luke/Documents/University/5th Year/Honours Python/Raw_Data/Hansen_Ames.csv")
hf_graph_dataframe = pd.read_pickle("C:/Users/Luke/Documents/University/5th Year/Honours Python/Transformed_Data/Hansen_HF_Graph.pkl")


def test_hf_graph_correct_atom_count() -> None:
    """
    Test function to check if the number of nodes in the graph matches the number of atoms in the molecule.

    This function selects a random index from the `hf_graph_dataframe` and retrieves the corresponding SMILES string.
    It then converts the SMILES string to a molecule object using `dm.to_mol` function.
    Finally, it compares the number of nodes in the graph with the number of atoms in the molecule and asserts their equality.

    Raises:
        AssertionError: If the number of nodes in the graph does not match the number of atoms in the molecule.
    """

    rand_idx = random.randint(0, len(hf_graph_dataframe) - 1)

    smiles = transformed_dataframe.iloc[rand_idx, 0]
    mol = dm.to_mol(smiles)

    graph = hf_graph_dataframe.iloc[rand_idx]

    assert graph.num_nodes == mol.GetNumAtoms()


def test_hf_graph_y_is_list() -> None:
    """
    Test function to check if the `y` column in the `hf_graph_dataframe` is a list.

    This function selects a random index from the `hf_graph_dataframe` and retrieves the corresponding `y` value.
    It then checks if the `y` value is a list and asserts the result.

    Raises:
        AssertionError: If the `y` value is not a list.
    """

    rand_idx = random.randint(0, len(hf_graph_dataframe) - 1)

    y = hf_graph_dataframe.iloc[rand_idx, 3]

    assert isinstance(y, list)


def test_graphormer_df_creator_truncated():
    from Data_Handling import graphormer_df_creator

    test_len = random.randint(1, 100)

    dataset = graphormer_df_creator("graphormer_format", truncated=True, truncated_length=test_len)

    assert dataset.num_rows == test_len
