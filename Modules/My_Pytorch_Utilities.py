from datetime import datetime
import torch
import torch.utils.data
import os
from torch.jit import script


class SMILES_Features_Dataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, pad: bool = False, pad_len: int = 2048) -> None:
        super().__init__()
        self.features = features
        self.labels = labels

        self.pad = pad
        self.pad_len = pad_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        labels = self.labels.iloc[idx]

        if self.pad is True:
            dim_padding = self.pad_len - (self.features.iloc[idx].size)
            features = torch.nn.functional.pad(torch.tensor(self.features.iloc[idx], dtype=torch.float32), (0, dim_padding))
            return torch.stack([features]), torch.tensor([labels], dtype=torch.float32)

        else:
            features = self.features.iloc[idx]
            return torch.tensor([features], dtype=torch.float32), torch.tensor([labels], dtype=torch.float32)


class Training_Utilities:
    def __init__(self):
        pass

    def reset_weights(self, model: torch.nn.Module | torch.nn.Sequential, verbose: bool = False) -> None:
        for layer in model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
                if verbose is False:
                    pass
                else:
                    print(f"Reset trainable parameters of layer = {layer}")


def metics_tensor_dict_to_floats(metrics):
    result = {}
    for key, value in metrics.items():
        if hasattr(value, "to"):
            result[key] = value.to(device="cpu", non_blocking=True).item()
        else:
            result[key] = value
    return result


def save(model: torch.nn.Module | torch.nn.Sequential, model_name: str, dummy_input: torch.Tensor) -> None:
    """_summary_

    Args:
        model (torch.nn.Module | torch.nn.Sequential): The callable instance of a PyTorch model
        model_name (str): Arbitrary name to describe the model
        dummy_input (torch.Tensor): Dummy tensor input to describe the shape of model input
    """
    date = datetime.now().strftime("%d-%m-%Y")
    timestamp = datetime.now().strftime("%H-%M_%d-%m-%Y")

    directory = f"Pytorch_Models/{date}"

    if not os.path.exists(directory):
        os.mkdir(directory)
        print(f"New directory created for today's PyTorch models: {directory}")
    else:
        print(f"Directory for today's PyTorch models already exists at {directory}. Saving current model here.")

    model_torch_script = script(model)
    model_torch_script.save(f"PyTorch_Models/{date}/{model_name}_{timestamp}.pt")
    torch.onnx.export(
        model,
        args=dummy_input,
        f=f"PyTorch_Models/{date}/{model_name}_{timestamp}.onnx",
    )
