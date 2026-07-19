import torch
import torch.nn as nn
import numpy as np

def get_batch(dataset, batch_size, context_length, device):
    rng = np.random.default_rng()
    upper_bound = len(dataset) - context_length
    start_indices = rng.integers(0 , upper_bound , batch_size)

    # shape: (batch_size, 1) + (context_length,) -> (batch_size, context_length)
    grid_indices = start_indices[:, None] + np.arange(context_length) #broadcasting 
    x_np = dataset[grid_indices]
    y_np = dataset[grid_indices + 1]
    x = torch.from_numpy(x_np).to(device, dtype=torch.long)
    y = torch.from_numpy(y_np).to(device, dtype=torch.long)
    return (x , y)



def save_checkpoint(model,optimizer,iteration,out):
   checkpoint = {}
   checkpoint["model"] = model.state_dict()
   checkpoint["optimizer"] = optimizer.state_dict()
   checkpoint["iteration"] = iteration 

   return torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer):
   checkpoint = torch.load(src , map_location='cpu')
   model.load_state_dict(checkpoint["model"])
   optimizer.load_state_dict(checkpoint["optimizer"])

   return checkpoint["iteration"] 






    

