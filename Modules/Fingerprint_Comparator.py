from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
import pandas as pd
from molfeat import trans
from molfeat.trans import pretrained
from sklearn import model_selection, ensemble
import torch
from Modules import PyTorch_Training, My_Pytorch_Utilities
import torchmetrics

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message="is_sparse is deprecated")


class Fingerprint_Comparator(ABC):
    def __init__(self, smiles: pd.Series | list[str], labels: pd.Series | list[str]) -> None:
        super().__init__()
        self.smiles = smiles
        self.labels = labels

    @abstractmethod
    def regular_fingerprint(self, fingerprints: list[str], k_folds: int = 10) -> pd.DataFrame:
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

    def __init__(
        self, smiles: pd.Series | list[str], labels: pd.Series | list[str], sklearn_classifier, scoring: str = "roc_auc"
    ) -> None:
        super().__init__(smiles, labels)
        self.sklearn_classifier = sklearn_classifier
        self.scoring = scoring

    def regular_fingerprint(self, fingerprints: list[str], k_folds: int = 10) -> pd.DataFrame:
        scores = []
        for fingerprint in tqdm(fingerprints):
            fp_transformer = trans.MoleculeTransformer(fingerprint)
            fingerprint_features = np.array(fp_transformer.transform(self.smiles))

            score = model_selection.cross_val_score(
                estimator=self.sklearn_classifier,
                X=fingerprint_features,
                y=self.labels,
                scoring=self.scoring,
                cv=model_selection.StratifiedKFold(k_folds, shuffle=True, random_state=42),
            )

            scores.append(score)
        return pd.DataFrame({"Fingerprint": fingerprints, f"Score ({self.scoring})": scores})

    def huggingface_fingerprint(self, fingerprints: list[str], k_folds: int = 10) -> pd.DataFrame:
        scores = []
        for fingerprint in tqdm(fingerprints):
            fp_transformer = pretrained.PretrainedHFTransformer(fingerprint)
            fingerprint_features = np.array(fp_transformer.transform(self.smiles))

            score = model_selection.cross_val_score(
                estimator=self.sklearn_classifier,
                X=fingerprint_features,
                y=self.labels,
                scoring=self.scoring,
                cv=model_selection.StratifiedKFold(k_folds, shuffle=True, random_state=42),
            )

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
            fingerprint_features: torch.Tensor = torch.tensor(
                fp_transformer.transform(self.smiles), dtype=torch.float32
            ).to(
                self.device
            )  # Transform SMILES > fingerprints

            dim_padding = torch.tensor(model_input_len) - fingerprint_features.size(
                1
            )  # Pad smaller fingerprints up to input size of model

            fingerprint_features = torch.nn.functional.pad(fingerprint_features, (0, dim_padding))

            with torch.no_grad():
                new_row = []
                for (
                    fp
                ) in (
                    fingerprint_features
                ):  # Run inference on each fingerprint, append the result to new row (representing a fingerprint methodology), calculate the accuracy of the prediction
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
            fingerprint_features: torch.Tensor = torch.tensor(
                fp_transformer.transform(self.smiles), dtype=torch.float32
            ).to(self.device)

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
    def __init__(
        self,
        smiles: pd.Series | list[str],
        labels: pd.Series | list[str],
        model: torch.nn.Module | torch.nn.Sequential,
        model_input_len: int,
        metric_collection: torchmetrics.MetricCollection,
    ) -> None:
        super().__init__(smiles, labels)
        self.labels = pd.Series(labels)
        self.model = model
        self.model_input_len = model_input_len

        self.metric_collection = metric_collection

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def regular_fingerprint(
        self,
        fingerprints: list[str] | pd.Series,
        pad: bool = False,
        loss_fn: torch.nn.Module = torch.nn.BCELoss(),
        optimizer=torch.optim.Adam,
        k_folds: int = 10,
        epochs: int = 10,
        batch_size: int = 256,
        DP: int = 3,
    ) -> tuple[pd.DataFrame, pd.MultiIndex]:
        score_df_list = []

        for fingerprint in fingerprints:
            print(f"{fingerprint:-^127}")

            fp_transformer = trans.MoleculeTransformer(fingerprint)

            fingerprint_features = fp_transformer.transform(self.smiles)

            if pad is False:  # Default behaviour - Train a new model for each FP type
                fp_len = len(fingerprint_features[0])
                self.model = PyTorch_Training.DILI_Models.DILI_Predictor_Sequential(fp_len, round(fp_len / 2)).to(
                    self.device
                )
                PyTorch_Dataset = My_Pytorch_Utilities.SMILES_Features_Dataset(
                    pd.Series(fingerprint_features), self.labels, pad=False, pad_len=self.model_input_len
                )

            else:  # Padding behaviour - Use pretrained model, pad to input len
                PyTorch_Dataset = My_Pytorch_Utilities.SMILES_Features_Dataset(
                    pd.Series(fingerprint_features), self.labels, pad=True, pad_len=self.model_input_len
                )

            loss, score_df = PyTorch_Training.Model_Train_Test(
                self.model, self.metric_collection, loss_fn, optimizer
            ).train_model_crossval(PyTorch_Dataset, k_folds, epochs, batch_size, DP)

            score_df.insert(0, "Fingerprint", fingerprint)  # Insert fingerprint type as first column
            # score_df.loc[len(score_df)-1] = means  # Calculate means for each fingerprint type
            # score_df = score_df.fillna({'Fingerprint': f'{fingerprint}_mean', 'Fold': 'NaN'})  # Add row title for means row

            score_df_list.append(score_df)  # Add the new score df to the list of score_dfs

        score_df: pd.DataFrame = pd.concat(score_df_list)
        score_df.set_index("Fingerprint")
        score_df_multiindex = pd.MultiIndex.from_frame(score_df)

        return score_df, score_df_multiindex

    def huggingface_fingerprint(self, fingerprints: list[str]) -> pd.DataFrame:
        return super().huggingface_fingerprint(fingerprints)
