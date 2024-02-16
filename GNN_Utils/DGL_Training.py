import dgl, dgl.data
import torch
from GNN_Utils.My_DGL_Utilities import batched_rand_sample_dataloader
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from GNN_Utils.My_DGL_Utilities import metric_plots


class DGL_Train_Test:
    def __init__(
        self,
        model,
        loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        lr=1e-3,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU

        self.model = model.to(self.device)  # The DGL Model

        self.loss_fn = loss_fn
        self.optimizer = optimizer(self.model.parameters(), lr=lr)

    def train_step(self, train_dataloader):
        self.model.train()
        train_loss = 0
        train_n_correct, train_num_tests = 0, 0

        for batched_graph, labels in train_dataloader:  # Enables dataloader batching
            batched_graph, labels = batched_graph.to(self.device), labels.to(self.device)
            pred_y = self.model(batched_graph, batched_graph.ndata["h"].float())

            train_n_correct += (pred_y.argmax(1) == labels).sum().item()
            train_num_tests += len(labels)

            loss = self.loss_fn(pred_y.squeeze(0), labels)
            train_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        train_loss = train_loss / len(train_dataloader)  # Average loss over the batches
        train_acc = train_n_correct / train_num_tests

        return train_loss, train_acc

    def test_step(self, test_dataloader):
        self.model.eval()
        test_loss = 0
        test_n_correct, test_num_tests = 0, 0

        with torch.inference_mode():  # Enables dataloader batching
            for batched_graph, labels in test_dataloader:
                batched_graph, labels = batched_graph.to(self.device), labels.to(self.device)
                pred_y = self.model(batched_graph, batched_graph.ndata["h"].float())

                test_n_correct += (pred_y.argmax(1) == labels).sum().item()
                test_num_tests += len(labels)

                loss = self.loss_fn(pred_y.squeeze(0), labels)
                test_loss += loss.item()

        test_loss = test_loss / len(test_dataloader)  # Average loss over the batches
        test_acc = test_n_correct / test_num_tests

        return test_loss, test_acc

    def train_model_crossval(self, dataset: dgl.data.DGLDataset, dataset_labels: pd.Series, epochs: int = 10, k_folds=10, DP=3):
        train_losses, test_losses = [], []
        test_accs = []

        kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

        for fold, (train_dataloader, test_dataloader) in enumerate(kfold.split(dataset, dataset_labels)):
            train_dataloader, test_dataloader = batched_rand_sample_dataloader(dataset)

            for epoch in range(epochs):
                test_loss, train_loss = [], []
                test_acc = []

                train_metrics = self.train_step(train_dataloader)
                test_metrics = self.test_step(test_dataloader)

                train_loss.append(train_metrics[0])
                test_loss.append(test_metrics[0])
                test_accs.append(test_metrics[1])

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            test_accs.append(test_acc)

            print(f"Fold {fold+1} results after {epochs} epochs: Test Accuracy: {train_metrics[1]:.{DP}f}")

        print(train_losses)
        metric_plots().sns_train_test_loss(train_losses, test_losses)
        # metric_plots().sns_train_test_acc(test_accs)

        # fig, ax = plt.subplots()
        # ax.plot(range(epochs), test_accs)
        # plt.show()
