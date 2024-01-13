import dgl, dgl.data
import torch
from GNN_Utils.My_DGL_Utilities import batched_rand_sample_dataloader

class DGL_Train_Test:
    def __init__(
        self,
        model,
        loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        lr=1e-4,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU

        self.model = model.to(self.device)  # The DGL Model

        self.loss_fn = loss_fn
        self.optimizer = optimizer(self.model.parameters(), lr=lr)

    def train_step(self, dataloader):
        self.model.train()
        train_loss = 0

        for batched_graph, labels in dataloader:  # Enables dataloader batching
            batched_graph, labels = batched_graph.to(self.device), labels.to(self.device)
            pred_y = self.model(batched_graph, batched_graph.ndata["h"].float())

            loss = self.loss_fn(pred_y.squeeze(0), labels)
            train_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        train_loss = train_loss / len(dataloader)  # Average loss over the batches

        return train_loss

    def test_step(self, dataloader):
        self.model.eval()
        test_loss = 0

        with torch.inference_mode():  # Enables dataloader batching
            for batched_graph, label in dataloader:
                batched_graph, label = batched_graph.to(self.device), label.to(self.device)
                pred_y = self.model(batched_graph, batched_graph.ndata["h"].float())

                loss = self.loss_fn(pred_y.squeeze(0), label)
                test_loss += loss.item()

        test_loss = test_loss / len(dataloader)  # Average loss over the batches

        return test_loss

    def train_model_crossval(self, dataset: dgl.data.DGLDataset, epochs: int = 10):
        train_dataloader, test_dataloader = batched_rand_sample_dataloader(dataset)

        for epoch in range(epochs):
            train_loss = self.train_step(train_dataloader)
            test_loss = self.test_step(test_dataloader)

            print(f"Epoch: {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}")
