import torch.nn as nn
import torch.nn.init as init

def reinitialize_specific_layers(model, param_names):
    """
    Re-initialize specific parameters by their full names, including .weight or .bias.
    For example: "time_embed.0.weight"

    :param model: The model instance (e.g., MultimodalUNet).
    :param param_names: A list of exact parameter names.
                        Example: ["time_embed.0.weight", "time_embed.0.bias"]
    """
    for param_name in param_names:
        # Split the parameter name to separate module path and param attribute
        parts = param_name.split('.')
        parent_module_path = '.'.join(parts[:-1])  # e.g. "time_embed.0"
        param_attribute = parts[-1]  # e.g. "weight" or "bias"

        try:
            # Get the parent module
            parent_module = model.get_submodule(parent_module_path)
        except AttributeError:
            print(f"Module {parent_module_path} not found for parameter {param_name}")
            continue

        # Now parent_module is something like nn.Linear, nn.Conv2d, nn.LayerNorm, etc.
        # Let's apply a suitable initialization based on the module type:
        if hasattr(parent_module, param_attribute):
            param_tensor = getattr(parent_module, param_attribute)

            # Apply initialization logic depending on the module type or parameter shape.
            # For example, if parent_module is a Linear layer and param_attribute is 'weight':
            if isinstance(parent_module, nn.Linear):
                if param_attribute == 'weight':
                    init.xavier_uniform_(param_tensor)
                elif param_attribute == 'bias' and param_tensor is not None:
                    init.zeros_(param_tensor)

            elif isinstance(parent_module, nn.Conv2d):
                if param_attribute == 'weight':
                    init.kaiming_normal_(param_tensor, nonlinearity='relu')
                elif param_attribute == 'bias' and param_tensor is not None:
                    init.zeros_(param_tensor)

            elif isinstance(parent_module, nn.LayerNorm) or isinstance(parent_module, nn.GroupNorm):
                # For LayerNorm, resetting to default initialization is often just ones and zeros:
                if param_attribute == 'weight':
                    nn.init.ones_(param_tensor)
                elif param_attribute == 'bias' and param_tensor is not None:
                    nn.init.zeros_(param_tensor)

            else:
                # If it's a type of module not explicitly handled, you can provide a generic init:
                if param_attribute == 'weight':
                    # Generic init if desired
                    init.xavier_uniform_(param_tensor)
                elif param_attribute == 'bias' and param_tensor is not None:
                    init.zeros_(param_tensor)
        else:
            print(f"Parameter {param_attribute} does not exist in module {parent_module_path}")

    print("Specified parameters have been re-initialized.")
