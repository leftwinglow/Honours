from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
import pandas as pd
from molfeat import trans
from molfeat.trans import pretrained
from sklearn import model_selection, ensemble
import torch
from typing import Union

sklearn_classifiers = Union[
    ensemble.AdaBoostClassifier,
    ensemble.BaggingClassifier,
    ensemble.ExtraTreesClassifier,
    ensemble.GradientBoostingClassifier,
    ensemble.HistGradientBoostingClassifier,
    ensemble.RandomForestClassifier,
    ensemble.StackingClassifier,
    ensemble.VotingClassifier
    ]


class Fingerprint_Comparator(ABC):
    def __init__(self, smiles, labels) -> None:
        super().__init__()
        self.smiles = smiles
        self.labels = labels

    @abstractmethod
    def regular_fingerprint_comparator(self, fingerprints: list[str]) -> pd.DataFrame:
        pass

    @abstractmethod
    def huggingface_fingerprint_comparator(self, fingerprints: list[str]) -> pd.DataFrame:
        pass


class Fingerprint_Comparator_SKlearn(Fingerprint_Comparator):
    """Data input for fingerprint comparison using scikit-learn classification models

    Parameters
    ----------
    smiles (pd.Series): The SMILES strings to be converted into fingerprints to act as the training set
    
    labels (pd.Series): Binary labels representing the DILI+ or DILI- nature of the SMILES strings
    
    sklearn_classifier (sklearn_classifiers): Any sklearn classification model
    
    scoring (str): An SKlearn classifier scoring method - https://scikit-learn.org/stable/modules/model_evaluation.html
    """
    def __init__(self, smiles: pd.Series, labels: pd.Series, sklearn_classifier: sklearn_classifiers, scoring: str) -> None:
        super().__init__(smiles, labels)
        self.sklearn_model = sklearn_classifier
        self.scoring = scoring

    def regular_fingerprint_comparator(self, fingerprints: list[str]) -> pd.DataFrame:
        scores = []
        for fingerprint in tqdm(fingerprints):
            fp_transformer = trans.MoleculeTransformer(featurizer=fingerprint)
            fingerprint_features: np.ndarray = np.array(fp_transformer.transform(self.smiles))

            score: int = model_selection.cross_val_score(estimator=self.sklearn_model, X=fingerprint_features, y=self.labels, scoring=self.scoring).mean()
            scores.append(score)
        return pd.DataFrame({"Fingerprint": fingerprints, f"Score ({self.scoring})": scores})

    def huggingface_fingerprint_comparator(self, fingerprints: list[str]) -> pd.DataFrame:
        scores = []
        for fingerprint in tqdm(fingerprints):
            fp_transformer = pretrained.PretrainedHFTransformer(kind=fingerprint)
            fingerprint_features: np.ndarray = np.array(fp_transformer.transform(self.smiles))

            score: int = model_selection.cross_val_score(estimator=self.sklearn_model, X=fingerprint_features, y=self.labels, scoring=self.scoring).mean()
            scores.append(score)
        return pd.DataFrame({"HuggingFace Fingerprint": fingerprints, f"Score ({self.scoring})": scores})


class Fingerprint_Comparator_Pytorch(Fingerprint_Comparator):
    """Data input for fingerprint comparison using scikit-learn classification models

    Parameters
    ----------
    smiles (pd.Series): The SMILES strings to be converted into fingerprints to act as the training set
    
    labels (pd.Series): Binary labels representing the DILI+ or DILI- nature of the SMILES strings
    """
    def __init__(self, smiles: pd.Series | str, labels: pd.Series, pytorch_model) -> None:
        super().__init__(smiles, labels)
        self.pytorch_model = pytorch_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def regular_fingerprint_comparator(self, fingerprints: list[str]) -> pd.DataFrame:
        scores = []
        for fingerprint in tqdm(fingerprints):
            fp_transformer = trans.MoleculeTransformer(featurizer=fingerprint)

            fingerprint_features = torch.tensor(fp_transformer.transform(self.smiles), dtype=torch.float32)

            fingerprint_features = fingerprint_features.to(self.device)
            print(fingerprint_features.shape)

            with torch.no_grad():
                self.pytorch_model.eval()
                output = self.pytorch_model(fingerprint_features)
                self.prediction = output.item()

        return pd.DataFrame({"Fingerprint": fingerprints, f"Prediction ({self.pytorch_model})": self.prediction})

    def huggingface_fingerprint_comparator(self, fingerprints: list[str]) -> pd.DataFrame:
        scores = []
        for fingerprint in tqdm(fingerprints):
            fp_transformer = pretrained.PretrainedHFTransformer(kind=fingerprint)
            
            fingerprint_features = torch.tensor(fp_transformer.transform(self.smiles), dtype=torch.float32)

            with torch.no_grad():
                self.pytorch_model.eval()
                output = self.pytorch_model(fingerprint_features)
                self.prediction = output.item()
                
        return pd.DataFrame({"Fingerprint": fingerprints, f"Prediction ({self.pytorch_model})": self.prediction})