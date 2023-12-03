from abc import ABC, abstractmethod
from pandas.core.api import DataFrame as DataFrame
from tqdm import tqdm
import numpy as np
import pandas as pd
from molfeat import trans
from molfeat.trans import pretrained
from sklearn import model_selection, ensemble
import torch
from typing import Union
from Modules import PyTorch_Training
import torchmetrics


sklearn_classifiers = Union[
    ensemble.AdaBoostClassifier,
    ensemble.BaggingClassifier,
    ensemble.ExtraTreesClassifier,
    ensemble.GradientBoostingClassifier,
    ensemble.HistGradientBoostingClassifier,
    ensemble.RandomForestClassifier,
    ensemble.StackingClassifier,
    ensemble.VotingClassifier,
]


class Fingerprint_Comparator(ABC):
    def __init__(self, smiles: pd.Series | list[str], labels: pd.Series | list[str]) -> None:
        super().__init__()
        self.smiles = smiles
        self.labels = labels

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

    sklearn_classifier (sklearn_classifiers): sklearn_classifiers

    scoring (str): An SKlearn classifier scoring method - https://scikit-learn.org/stable/modules/model_evaluation.html
    """

    def __init__(self, smiles: pd.Series | list[str], labels: pd.Series | list[str], sklearn_classifier: sklearn_classifiers, scoring: str = "roc_auc") -> None:
        super().__init__(smiles, labels)
        self.sklearn_classifier = sklearn_classifier
        self.scoring = scoring

    def regular_fingerprint(self, fingerprints: list[str]) -> pd.DataFrame:
        scores = []
        for fingerprint in tqdm(fingerprints):
            fp_transformer = trans.MoleculeTransformer(fingerprint)
            fingerprint_features = np.array(fp_transformer.transform(self.smiles))

            score: int = model_selection.cross_val_score(
                estimator=self.sklearn_classifier,
                X=fingerprint_features,
                y=self.labels,
                scoring=self.scoring,
            ).mean()
            
            scores.append(score)
        return pd.DataFrame({"Fingerprint": fingerprints, f"Score ({self.scoring})": scores})

    def huggingface_fingerprint(self, fingerprints: list[str]) -> pd.DataFrame:
        scores = []
        for fingerprint in tqdm(fingerprints):
            fp_transformer = pretrained.PretrainedHFTransformer(fingerprint)
            fingerprint_features = np.array(fp_transformer.transform(self.smiles))

            score: int = model_selection.cross_val_score(
                estimator=self.sklearn_classifier,
                X=fingerprint_features,
                y=self.labels,
                scoring=self.scoring,
            ).mean()
            
            scores.append(score)
        return pd.DataFrame({"HuggingFace Fingerprint": fingerprints, f"Score ({self.scoring})": scores})


class PyTorch_Pretrained(Fingerprint_Comparator):
    """Data input for fingerprint comparison using a pretrained PyTorch model

    Key Question: What fingerprinting predictions are yielded on SMILES across different fingerprint types

    Parameters
    ----------
    smiles (pd.Series): The SMILES strings to be converted into fingerprints test the model on

    labels (pd.Series): Binary labels representing the DILI+ or DILI- nature of the SMILES strings

    pytorch_model (_type_): _description_
    """

    def __init__(self, smiles, labels, pretrained_model: torch.nn.Module) -> None:
        super().__init__(smiles, labels)
        self.pretrained_model = pretrained_model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def b_acc(self, y_true, y_pred):
        correct = sum(1 for true, pred in zip(y_true, y_pred) if (pred >= 0.5) == true)
        total = len(y_true)
        binary_accuracy = correct / total
        return binary_accuracy

    def regular_fingerprint(self, fingerprints: list[str], model_input_len: int) -> pd.DataFrame:
        """Compare the performance of a pretrained PyTorch model using different molfeat regular fingerprints

        Args:
            fingerprints (list[str]): A list of fingerprints to compare model performance on
            model_input_len (int): The length of fingerprint the model was trained on (e.g. if the model was trained on ECFP, 2048 should be given)

        Returns:
            pd.DataFrame: A pandas dataframe showing the fingerprinting methodology and the scores produced by them
        """
        score_list = []
        for fingerprint in tqdm(fingerprints):
            fp_transformer = trans.MoleculeTransformer(fingerprint)
            fingerprint_features: torch.Tensor = torch.tensor(fp_transformer.transform(self.smiles), dtype=torch.float32).to(self.device)  # Transform SMILES > fingerprints

            dim_padding = torch.tensor(model_input_len) - fingerprint_features.size(1)  # Pad smaller fingerprints up to input size of model

            fingerprint_features = torch.nn.functional.pad(fingerprint_features, (0, dim_padding))

            with torch.no_grad():
                new_row = []
                for fp in fingerprint_features:  # Run inference on each fingerprint, append the result to new row (representing a fingerprint methodology), calculate the accuracy of the prediction
                    output = self.pretrained_model(fp)
                    new_row.append(output.item())
                score_list.append(self.b_acc(self.labels, new_row))

        score_dataframe = pd.DataFrame(score_list, columns=["acc"])
        score_dataframe.insert(0, "Fingerprint type", fingerprints)
        return score_dataframe

    def huggingface_fingerprint(self, fingerprints: list[str]) -> pd.DataFrame:
        score_list = []
        for fingerprint in tqdm(fingerprints):
            fp_transformer = pretrained.PretrainedHFTransformer(fingerprint)
            fingerprint_features: torch.Tensor = torch.tensor(fp_transformer.transform(self.smiles), dtype=torch.float32).to(self.device)

            dim_padding = torch.tensor(2048) - fingerprint_features.size(1)

            with torch.no_grad():
                new_row = []
                for fp in fingerprint_features:
                    output = self.pretrained_model(fp)
                    new_row.append(output.item())
                score_list.append(self.b_acc(self.labels, new_row))

        score_dataframe = pd.DataFrame(score_list, columns=["acc"])
        score_dataframe.insert(0, "Fingerprint type", fingerprints)
        return score_dataframe

class Pytorch_Train(Fingerprint_Comparator):
    def __init__(self, smiles: pd.Series | list[str], labels: pd.Series | list[str], model: torch.nn.Module | torch.nn.Sequential, model_input_len: int, metric_collection: torchmetrics.MetricCollection) -> None:
        super().__init__(smiles, labels)
        self.model = model        
        self.model_input_len = model_input_len
        
        self.metric_collection = metric_collection
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def regular_fingerprint(self, fingerprints: list[str]) -> pd.DataFrame:
        for fingerprint in tqdm(fingerprints):
            fp_transformer = trans.MoleculeTransformer(fingerprint)
            # fingerprint_features = torch.tensor(fp_transformer.transform(self.smiles), dtype=torch.float32).to(self.device)
            fingerprint_features = fp_transformer.transform(self.smiles)
            
            PyTorch_Dataset = PyTorch_Training.SMILES_Features_Dataset(fingerprint_features, self.labels)
            
            # dim_padding = torch.tensor(self.model_input_len) - fingerprint_features.size
            
            # fingerprint_features = torch.nn.functional.pad(fingerprint_features, (0, dim_padding))
            
            loss, score_df = PyTorch_Training.Model_Train_Test(self.model, self.metric_collection).train_model_crossval(PyTorch_Dataset)
    
        return score_df
    
    def huggingface_fingerprint(self, fingerprints: list[str]) -> DataFrame:
        return super().huggingface_fingerprint(fingerprints)