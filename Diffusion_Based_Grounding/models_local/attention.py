import numpy as np
import torch
from generative.networks.nets.diffusion_model_unet import CrossAttention



# This function will be used to apply the desired modifications to each CrossAttention layer
def add_attention_scores_list_attr(cross_attention_layer):
    cross_attention_layer.attention_scores_list = []
    
def _attention_new(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    dtype = query.dtype
    if self.upcast_attention:
        query = query.float()
        key = key.float()

    attention_scores = torch.baddbmm(
        torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
        query,
        key.transpose(-1, -2),
        beta=0,
        alpha=self.scale,
    )
    
    attention_probs = attention_scores.softmax(dim=-1)
    attention_probs = attention_probs.to(dtype=dtype)
    self.attention_scores_list.append(attention_probs.cpu())

    x = torch.bmm(attention_probs, value)
    return x

# Recursive function to walk through the model's modules and modify CrossAttention layers
def modify_cross_attention_layers(module):
    for child_name, child_module in module.named_children():
        if isinstance(child_module, CrossAttention):
            add_attention_scores_list_attr(child_module)
            child_module._attention = _attention_new.__get__(child_module, CrossAttention)
        else:
            # Recursively apply modifications to children modules
            modify_cross_attention_layers(child_module)
            
# Assuming 'DiffusionModelUNet' is your model's class and 'model' is its instance
# Define a recursive function to collect layers
def find_cross_attention_layers(module, layers=None):
    if layers is None:
        layers = []

    # Iterate over all modules in the current module
    for name_mod, mod in module.named_children():
        # If the module is a CrossAttention instance, add it to the list
        if isinstance(mod, CrossAttention) and name_mod == 'attn2':
            layers.append(mod)
        else:
            # Else, recursively search in children modules
            find_cross_attention_layers(mod, layers)
    return layers

def get_spatial_attention_map(model, layer_idx, timesteps_idx):
    cross_attention_layers = find_cross_attention_layers(model)
    attention_map = cross_attention_layers[layer_idx].attention_scores_list[timesteps_idx]
    shape_init = attention_map.shape
    spatial_dim = int(np.sqrt(shape_init[1]))
    attention_map_spatial = attention_map.reshape(shape_init[0],spatial_dim,spatial_dim,shape_init[2])
    return attention_map_spatial