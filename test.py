import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from Modules import Fingerprint_Comparator
from sklearn import ensemble
import torch


tox_df = pd.read_csv("Transformed_Data/Final_DILI.csv")
pytorch_model = torch.load("PyTorch_Models/12-11-2023/DILIst_21-53_12-11-2023.pt")

fp = ["ecfp"]

def sklearn_comparator(tox_df, fp):
    model_rf = ensemble.GradientBoostingClassifier(random_state=42)
    sklearn_test = Fingerprint_Comparator.SKlearn(scoring="roc_auc",smiles=tox_df["SMILES"],labels=tox_df["DILI?"], sklearn_classifier=model_rf).regular_fingerprint(fingerprints=fp)
    
    return sklearn_test

def pytorch_comparator(fp):
    pytorch_smiles = ["CCC", "Fc1ccc(cc1)[C@@]3(OCc2cc(C#N)ccc23)CCCN(C)C"]
    pytorch_test = Fingerprint_Comparator.PyTorch_Pretrained(smiles=pytorch_smiles, pretrained_model=pytorch_model).regular_fingerprint(fingerprints=fp)

    return pytorch_test

# print(sklearn_comparator(tox_df, fp))
# print(pytorch_comparator(fp))


validation_smiles = ["CCC", "CCCC", "Fc1ccc(cc1)[C@@]3(OCc2cc(C#N)ccc23)CCCN(C)C", "CCCCC"]

print(Fingerprint_Comparator.PyTorch_Pretrained(validation_smiles, pytorch_model).regular_fingerprint(["ecfp", "ecfp-count"]))