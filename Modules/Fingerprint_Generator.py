import pandas as pd
from molfeat import trans

def generate_fp_column(dataframe, dataframe_smiles_col, fp_type: str) -> pd.DataFrame:
    """Adds a column of SMILES strings with Molfeat fingerprints

    Args:
        dataframe (_type_): The dataframe object
        dataframe_smiles_col (_type_): The SMILES column of the dataframe object
        fp_type (str): Molfeat fingerprint type

    Returns:
        pd.DataFrame: Dataframe with a column of user-defined Molfeat fingerprints
    """
    fp_transformer = trans.MoleculeTransformer(featurizer=f'{fp_type}')
    dataframe[f"{fp_type}"] = fp_transformer.transform(dataframe_smiles_col.values)
    return dataframe
