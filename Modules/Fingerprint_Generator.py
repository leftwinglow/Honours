from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import torch
from molfeat import trans, store
from molfeat.trans import pretrained
from typing import TypeVar


class Fingerprint_Lists:
    def verbose_checker(self, verbose, fingerprints):
        if verbose is True:
            print(f"Fingerprint types: {fingerprints}")
        else:
            pass

    def abridged_set(self, fingerprints: list[str], abridged_set: bool, abridged_count: int):
        if abridged_set is False:
            pass
        else:
            fingerprints = fingerprints[:abridged_count]

    def regular_fingerprints(self, abridged_set: bool = False, abridged_count: int = 3, verbose: bool = False) -> list[str]:
        fingerprints = [
            "maccs",
            # "mordred",
            "ecfp",
            "ecfp-count",
            "avalon",
            "fcfp",
            "secfp",
            "topological",
            "atompair",
            "rdkit",
            "pattern",
            "layered",
        ]

        self.abridged_set(fingerprints, abridged_set, abridged_count)

        self.verbose_checker(verbose, fingerprints)

        return fingerprints

    def huggingface_fingerprints(self, abridged_set: bool = False, abridged_count: int = 3, verbose: bool = False) -> list[str]:
        fingerprints = [
            "ChemGPT-4.7M",
            "ChemGPT-19M",
            "GPT2-Zinc480M-87M",
            "MolT5",
        ]

        self.abridged_set(fingerprints, abridged_set, abridged_count)

        self.verbose_checker(verbose, fingerprints)

        return fingerprints


def generate_fp_column(dataframe: pd.DataFrame, dataframe_smiles_col: pd.Series, fp_type: str) -> pd.DataFrame:
    """Adds a column of SMILES strings with Molfeat fingerprints

    Args:
        dataframe (_type_): The dataframe object
        dataframe_smiles_col (_type_): The SMILES column of the dataframe object
        fp_type (str): Molfeat fingerprint type

    Returns:
        pd.DataFrame: Dataframe with a column of user-defined Molfeat fingerprints
    """
    fp_transformer = trans.MoleculeTransformer(featurizer=f"{fp_type}")
    dataframe[f"{fp_type}"] = fp_transformer.transform(dataframe_smiles_col.values)
    return dataframe


class Smiles_To_Fingerprint:
    """Convert a SMILES string to a fingerprint of a given datatype

    Returns:
        _type_: Fingerprint as an array, list, tensor, etc
    """

    T = TypeVar("T")

    def __init__(self, smiles: list[str], fp_type: str) -> None:
        fp_transformer = trans.MoleculeTransformer(featurizer=fp_type)
        self.fingerprint = fp_transformer.transform(smiles)

    def astype(self, return_type: type[T]) -> T:
        match return_type:
            case list() | tuple():
                pass
            case np.ndarray:
                self.fingerprint = np.array(self.fingerprint)
            case torch.Tensor:
                self.fingerprint = torch.tensor(self.fingerprint, dtype=torch.float32)
            case _:
                raise TypeError(f"Unsupported type: '{type}'")
        return self.fingerprint
