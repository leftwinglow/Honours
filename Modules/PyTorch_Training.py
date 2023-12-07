import pandas as pd
import torch.nn as nn
import torch.utils.data
import torchmetrics
from Modules import My_Pytorch_Utilities
from sklearn.model_selection import StratifiedKFold

class DILI_Models:  
    class DILI_Predictor_Sequential(nn.Sequential):
        def __init__(self, input_size, hidden_size, output_size) -> None:
            super().__init__(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size),
                nn.Sigmoid()
            )

class Model_Train_Test:
    def __init__(self, model: torch.nn.Module, metric_collection: torchmetrics.MetricCollection, loss_fn: torch.nn.Module = nn.BCELoss(), optimizer=torch.optim.Adam, lr=1e-4) -> None:
        self.model = model  # The PyTorch Model
        self.metric_collection = metric_collection  # A metric collection, on device

        self.loss_fn = loss_fn
        self.optimizer = optimizer(self.model.parameters(), lr=lr)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU

    def train_step(self, dataloader):
        self.model.train()
        train_loss = 0
        train_metrics = self.metric_collection.clone(prefix="train_")

        for batch, (X, y) in enumerate(dataloader):  # Enables dataloader batching
            X, y = X.to(self.device), y.to(self.device)
            pred_y = self.model(X).squeeze(1)

            loss = self.loss_fn(pred_y, y)
            train_loss += loss.item()
            train_metrics_results = train_metrics(pred_y, y)  # Calculate TorchMetrics

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        train_loss = train_loss / len(dataloader)  # Average loss over the batches

        train_metrics_results = train_metrics.compute() # "Average" TorchMetrics over the batches
        train_metrics.reset()

        return train_loss, train_metrics_results

    def test_step(self, dataloader):
        self.model.eval()
        test_loss = 0
        test_metrics = self.metric_collection.clone(prefix="test_")

        with torch.inference_mode():  # Enables dataloader batching
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)
                pred_y = self.model(X).squeeze(1)

                loss = self.loss_fn(pred_y, y)
                test_loss += loss.item()
                test_metrics_results = test_metrics(pred_y, y)  # Calculate TorchMetrics

            test_metrics_results = test_metrics.compute()  # "Average" TorchMetrics over the batches
            test_metrics.reset()

        test_loss = test_loss / len(dataloader)  # Average loss over the batches

        return test_loss, test_metrics_results

    def train_model_crossval(self, dataset: torch.utils.data.Dataset, k_folds: int = 10, epochs: int = 10, batch_size: int = 256, DP: int = 3):
        score_df = []
        loss = {"train_loss": [], "test_loss": []}

        kfold = StratifiedKFold(k_folds, shuffle=True, random_state=42)

        for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset, dataset.labels)):
            
            train_dataset = torch.utils.data.Subset(dataset, train_idx)
            test_dataset = torch.utils.data.Subset(dataset, test_idx)

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=True)

            My_Pytorch_Utilities.Training_Utilities().reset_weights(self.model)  # Reset the model weights between each fold

            for epoch in range(epochs):
                train_loss, train_metrics = self.train_step(train_loader)
                test_loss, test_metrics = self.test_step(test_loader)

                loss["train_loss"].append(train_loss)
                loss["test_loss"].append(test_loss)

                train_metrics, test_metrics = My_Pytorch_Utilities.metics_tensor_dict_to_floats(train_metrics), My_Pytorch_Utilities.metics_tensor_dict_to_floats(test_metrics)  # Convert the dictionary with tensors on GPU to a dictionary with floats on CPU

            print(f"Fold {fold+1} final results after {epoch+1} epochs: Train Acc: {train_metrics['train_BinaryAccuracy']:.{DP}f} Train Loss: {train_loss:.{DP}f} (n = {len(train_idx)}) | Test Acc: {test_metrics['test_BinaryAccuracy']:.{DP}f} Test Loss: {test_loss:.{DP}f} (n = {len(test_idx)}) ")

            score_df.append(pd.DataFrame.from_dict(test_metrics, orient="index").transpose().round(DP))  # Add df to the list of dfs

        score_df = pd.concat(score_df)  # Concatenate the dfs - More performant
        score_df.insert(0, "Fold", range(k_folds))

        return loss, score_df

    def train_model(self, dataset, train_size: float = 0.8, epochs: int = 10, batch_size: int = 256, DP: int = 3):
        score_df = []
        loss = {"train_loss": [], "test_loss": []}

        train_size = int(len(dataset) * 0.8)
        validation_size = len(dataset) - train_size

        train_data, validation_data = torch.utils.data.random_split(dataset, [train_size, validation_size])
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(validation_data, batch_size, shuffle=True)

        for epoch in range(epochs):
            train_loss, train_metrics = self.train_step(train_loader)
            test_loss, test_metrics = self.test_step(validation_loader)

            loss["train_loss"].append(train_loss)
            loss["test_loss"].append(test_loss)

            train_metrics, test_metrics = My_Pytorch_Utilities.metics_tensor_dict_to_floats(train_metrics), My_Pytorch_Utilities.metics_tensor_dict_to_floats(test_metrics)  # Convert the dictionary with tensors on GPU to a dictionary with floats on CPU

            print(f"Results after {epoch+1} epochs: Train Acc: {train_metrics['train_BinaryAccuracy']:.{DP}f} Train Loss: {train_loss:.{DP}f} (n = {len(train_data)}) | Test Acc: {test_metrics['test_BinaryAccuracy']:.{DP}f} Test Loss: {test_loss:.{DP}f} (n = {len(validation_data)}) ")

            score_df.append(pd.DataFrame.from_dict(test_metrics, orient="index").transpose().round(DP))  # Add df to the list of dfs

        score_df = pd.concat(score_df)  # Concatenate the dfs - More performant
        score_df.insert(0, "Epoch", range(epochs))
        score_df.set_index("Epoch", inplace=True)

        return loss, score_df
