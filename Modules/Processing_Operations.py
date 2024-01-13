import pandas as pd
import pubchempy as pcp
import numpy as np
import datamol as dm
import gc


def replace_drug_name_with_smiles(dataframe: pd.DataFrame, drug_name_col: str) -> pd.DataFrame:
    """Convert generic or IUPAC drug names to SMILES strings

    Args:
        dataframe (pd.DataFrame): A pandas dataframe
        drug_name_col (str): Title of the dataframe column containing the drug names which should be converted to SMILES strings

    Returns:
        pd.DataFrame: _description_
    """
    dataframe[drug_name_col] = dataframe[drug_name_col].map(
        lambda x: pcp.get_compounds(identifier=x, namespace="name")
    )  # Get pubchem CID for each compound
    dataframe = dataframe[dataframe[drug_name_col].map(lambda d: len(d)) == 1]  # Drop columns with multiple chemical identifiers
    dataframe[drug_name_col] = dataframe[drug_name_col].str[0]  # Convert list of pubchempy compounds to str
    dataframe[drug_name_col] = dataframe[drug_name_col].apply(lambda x: x.isomeric_smiles)  # Get isomeric smiles for pubchempy compounds
    return dataframe


def get_coordinates(mol, parallel: bool = False) -> list[np.ndarray]:
    """Returns the coordinates of a molecule"""
    if parallel is True:
        conformer = dm.utils.parallelized(fn=(dm.conformers.generate), inputs_list=mol, progress=True, batch_size=12)
        coordinates = []
        for i, mol in enumerate(conformer):
            coordinates.append(conformer[i].GetConformer().GetPositions())
        gc.collect()
    else:
        conformer = dm.conformers.generate(mol, align_conformers=True, minimize_energy=False, num_threads=8)
        coordinates = conformer.GetConformer().GetPositions()

    return coordinates
