import mlx.nn as nn
import mlx.core as mx

def get_submodule(model:nn.Module, path:str):
    #NOTE: This is required since tree flattening is not possible thanks to switching sequentials
    # Sacrificing inbuilt functions for code cleanliness, separation of concerns and understanding
    parts = path.split('.')
    if(parts[-1] == 'weight' or parts[-1] == 'bias'):
        parts.pop()
    current = model
    
    for part in parts:
        if part.isdigit():
            idx = int(part)
            if hasattr(current, '__getitem__'):
                current = current[idx]
            elif hasattr(current, "layers"):
                current = current.layers[idx]
            else:
                raise ValueError(f"Cannot index into {type(current)} with {idx}")
        else:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                if hasattr(current, 'parameters') and callable(current.parameters):
                    params = current.parameters()
                    if part in params:
                        return params[part]
                raise AttributeError(f"'{type(current).__name__}' has no attribute '{part}'")
    return current

def get_query_projection(name:str):
    proj = 'k' if 'k_proj' in name else 'q' if 'q_proj' in name else 'v' if 'v_proj' in name else None
    if(proj is None):
        proj = 'k' if 'to_k' in name else 'q' if 'to_q' in name else 'v' if 'to_v' in name else None
    return proj

def get_flattened_state_dict(model:nn.Module):
    flat_dict = {}
    
    def _flatten(prefix, params):
        for k, v in params.items():
            name = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _flatten(name, v)
            else:
                flat_dict[name] = v
    
    _flatten("", model.parameters())
    return flat_dict
