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
    def __init__(self, smiles) -> None:
        super().__init__()
        self.smiles = smiles

    @abstractmethod
    def regular_fingerprint(self, fingerprints: list[str]) -> pd.DataFrame:
        pass

    @abstractmethod
    def huggingface_fingerprint(self, fingerprints: list[str]) -> pd.DataFrame:
        pass


class SKlearn(Fingerprint_Comparator):
    """Data input for fingerprint comparison using scikit-learn classification models
    
    Key Question: What fingerprinting methodologies trains the most accurate scikit learn model?

    Perform cross-validation on a given sklearn model to determine which fingerprint method produces the greatest score

    Parameters
    ----------
    smiles (pd.Series): The SMILES strings to be converted into fingerprints to act as the training set
    
    labels (pd.Series): Binary labels representing the DILI+ or DILI- nature of the SMILES strings
    
    sklearn_classifier (sklearn_classifiers): Any sklearn classification model
    
    scoring (str): An SKlearn classifier scoring method - https://scikit-learn.org/stable/modules/model_evaluation.html
    """
    def __init__(self, smiles: pd.Series, labels: pd.Series, sklearn_classifier: sklearn_classifiers, scoring: str) -> None:
        super().__init__(smiles)
        self.sklearn_model = sklearn_classifier
        self.labels = labels
        self.scoring = scoring

    def regular_fingerprint(self, fingerprints: list[str]) -> pd.DataFrame:
        scores = []
        for fingerprint in tqdm(fingerprints):
            fp_transformer = trans.MoleculeTransformer(featurizer=fingerprint)
            fingerprint_features: np.ndarray = np.array(fp_transformer.transform(self.smiles))

            score: int = model_selection.cross_val_score(estimator=self.sklearn_model, X=fingerprint_features, y=self.labels, scoring=self.scoring).mean()
            scores.append(score)
        return pd.DataFrame({"Fingerprint": fingerprints, f"Score ({self.scoring})": scores})

    def huggingface_fingerprint(self, fingerprints: list[str]) -> pd.DataFrame:
        scores = []
        for fingerprint in tqdm(fingerprints):
            fp_transformer = pretrained.PretrainedHFTransformer(kind=fingerprint)
            fingerprint_features: np.ndarray = np.array(fp_transformer.transform(self.smiles))

            score: int = model_selection.cross_val_score(estimator=self.sklearn_model, X=fingerprint_features, y=self.labels, scoring=self.scoring).mean()
            scores.append(score)
        return pd.DataFrame({"HuggingFace Fingerprint": fingerprints, f"Score ({self.scoring})": scores})


class PyTorch_Pretrained(Fingerprint_Comparator):
    """Data input for fingerprint comparison using a pretrained PyTorch model

    Key Question: What fingerprinting methodologies yields the best accuracy on a pretrained PyTorch model?

    Parameters
    ----------
    smiles (pd.Series): The SMILES strings to be converted into fingerprints test the model on
    
    labels (pd.Series): Binary labels representing the DILI+ or DILI- nature of the SMILES strings
    
    pytorch_model (_type_): _description_
    """
    def __init__(self, smiles: pd.Series | list[str], pretrained_model: torch.nn.Module) -> None:
        super().__init__(smiles)
        self.pretrained_model = pretrained_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def regular_fingerprint(self, fingerprints: list[str]) -> pd.DataFrame:
        score_list = []
        
        for fp_type in tqdm(fingerprints):
            fp_transformer = trans.MoleculeTransformer(featurizer=fp_type)
            fingerprint_features: torch.Tensor = torch.tensor(fp_transformer.transform(self.smiles), dtype=torch.float32).to(self.device)

            with torch.no_grad():
                new_row = []
                self.pretrained_model.eval()
                for fp in fingerprint_features:
                    output = self.pretrained_model(fp)
                    self.prediction = output.item()
                    new_row.append(self.prediction)
                score_list.append(new_row)

        df = pd.DataFrame(score_list, columns=self.smiles)
        df.insert(0, "Fingerprint type", fingerprints) 
        return df

    def huggingface_fingerprint(self, fingerprints: list[str]) -> pd.DataFrame:
        scores = []
        for fingerprint in tqdm(fingerprints):
            fp_transformer = pretrained.PretrainedHFTransformer(kind=fingerprint)
            fingerprint_features: torch.Tensor = torch.tensor(fp_transformer.transform(self.smiles), dtype=torch.float32).to(self.device)

            with torch.no_grad():
                self.pretrained_model.eval()
                output = self.pretrained_model(fingerprint_features)
                self.prediction = output.item()
                
        return pd.DataFrame({"Fingerprint": fingerprints, f"Prediction ({self.pretrained_model})": self.prediction})