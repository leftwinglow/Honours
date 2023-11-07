import pandas as pd
import pubchempy as pcp

def replace_drug_name_with_smiles(dataframe: pd.DataFrame, drug_name_col: str) -> pd.DataFrame:
    """Convert generic or IUPAC drug names to SMILES strings

    Args:
        dataframe (pd.DataFrame): A pandas dataframe
        drug_name_col (str): Title of the dataframe column containing the drug names which should be converted to SMILES strings

    Returns:
        pd.DataFrame: _description_
    """
    dataframe[drug_name_col] = dataframe[drug_name_col].map(lambda x: pcp.get_compounds(identifier=x, namespace='name')) # Get pubchem CID for each compound
    dataframe = dataframe[dataframe[drug_name_col].map(lambda d: len(d)) == 1] # Drop columns with multiple chemical identifiers
    dataframe[drug_name_col] = dataframe[drug_name_col].str[0] # Convert list of pubchempy compounds to str
    dataframe[drug_name_col] = dataframe[drug_name_col].apply(lambda x: x.isomeric_smiles) # Get isomeric smiles for pubchempy compounds
    return dataframe