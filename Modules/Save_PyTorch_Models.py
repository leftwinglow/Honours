from datetime import datetime
import torch
import os

def save(model, model_name: str, dummy_input: torch.Tensor) -> None:
    """_summary_

    Args:
        model (_type_): The callable instance of a PyTorch model
        model_name (str): Arbitrary name to describe the model 
        dummy_input (torch.Tensor): Dummy tensor input to describe the shape of model input
    """
    date = datetime.now().strftime("%d-%m-%Y")
    timestamp = datetime.now().strftime('%H-%M_%d-%m-%Y')
    
    directory = f"Pytorch_Models/{date}"
    
    if not os.path.exists(directory):
        os.mkdir(directory)
        print(f"New directory created for today's PyTorch models: {directory}")
    else:
        print(f"Directory for today's PyTorch models already exists at {directory}. Placing saved modles in here.")
    
    model_torch_script = torch.jit.script(model)
    model_torch_script.save(f"PyTorch_Models/{date}/{model_name}_{timestamp}.pt")
    torch.onnx.export(model, args=dummy_input, f=f"PyTorch_Models/{date}/{model_name}_{timestamp}.onnx")