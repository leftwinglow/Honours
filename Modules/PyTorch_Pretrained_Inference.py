from Modules.Fingerprint_Comparator import Fingerprint_Comparator

import pandas as pd
import torch
from molfeat import trans
from molfeat.trans import pretrained
from tqdm import tqdm


class DILI_Inference(Fingerprint_Comparator):
    """Data input for fingerprint comparison using a pretrained PyTorch model

    Key Question: What fingerprinting predictions are yielded on SMILES across different fingerprint types

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
                    new_row.append(output.item())
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

        return pd.DataFrame(
            {
                "Fingerprint": fingerprints,
                f"Prediction ({self.pretrained_model})": self.prediction,
            }
        )
