import pandas as pd
from Modules import Fingerprint_Comparator
from sklearn import ensemble
import torch

model0 = torch.load("PyTorch_Models/12-11-2023/DILIst_21-53_12-11-2023.pt")

fp = ["ecfp"]
tox_df = pd.read_csv("Transformed_Data/Final_DILI.csv")
# model_rf = ensemble.GradientBoostingClassifier(random_state=42)

# test = Fingerprint_Comparator.Fingerprint_Comparator_SKlearn(scoring="roc_auc",smiles=tox_df["SMILES"],labels=tox_df["DILI?"], sklearn_classifier=model_rf).regular_fingerprint_comparator(fingerprints=fp)

# print(test)
validation_smiles = "CCC"
pytorch_test = Fingerprint_Comparator.Fingerprint_Comparator_Pytorch(smiles=validation_smiles,labels=tox_df["DILI?"], pytorch_model=model0).regular_fingerprint_comparator(fingerprints=fp)

print(pytorch_test)