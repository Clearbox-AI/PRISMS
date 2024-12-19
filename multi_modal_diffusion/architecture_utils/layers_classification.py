from collections import defaultdict

ONLY_TAB = "only_tabular"
ONLY_IMG = "only_image"
COMMON = "common"

def get_param_ancestry(model):
    """
    Recursively traverse the model and record the classes of modules each parameter belongs to.
    Returns a dict:
    {
        "param_name": [list_of_ancestor_module_classes],
        ...
    }
    """
    param_ancestry = {}

    def recurse(module, prefix, ancestors):
        for name, child in module.named_children():
            # Extend the ancestor list with child's class
            child_ancestors = ancestors + [child.__class__]
            child_prefix = f"{prefix}.{name}" if prefix else name
            recurse(child, child_prefix, child_ancestors)

        # Record parameter ancestry at this level
        for pname, p in module.named_parameters(recurse=False):
            full_name = f"{prefix}.{pname}" if prefix else pname
            param_ancestry[full_name] = ancestors

    recurse(model, prefix="", ancestors=[model.__class__])
    return param_ancestry

def classify_parameters(model, debug=False):
    """
    Classify parameters into 'only-tabular', 'only-image', or 'common'.

    Rules:
    - If any ancestor of the parameter is CrossAttentionBlock or QKVAttention,
      then the parameter is 'common'.
    - Else, if matches tabular substrings in the name: 'only-tabular'
    - Else, if matches image substrings in the name: 'only-image'
    - Else 'common'.
    """

    # Identify classes that define 'common' by structural logic
    from multi_modal_diffusion.model.mm_unet import CrossAttentionBlock, QKVAttention

    layer_classification = defaultdict(list)

    # Define heuristics for tabular and image
    tabular_substrings = [
        "tabular_in_layers",
        "tabular_out_layers",
        "tabular_skip_connection",
        "tabular_out",
        "tabular_mlp",
        "tab_upd",
        "tab_norm",
        "tab_qkv",
        "tab_proj_out",
        "tabular_attention_block",
    ]

    image_substrings = [
        "image_in_layers",
        "image_out_layers",
        "image_skip_connection",
        "image_out",
        "image_conv",
        "img_upd",
        "img_norm",
        "img_qkv",
        "img_proj_out",
        "image_attention_block",
    ]

    # Get param ancestry
    param_ancestry = get_param_ancestry(model)

    for name, param in model.named_parameters():
        ancestors = param_ancestry[name]

        # Check if in cross-attention or QKVAttention subtree
        if any(cls in [CrossAttentionBlock, QKVAttention] for cls in ancestors):
            category = COMMON
        else:
            # Not in a cross-attention subtree, apply name-based heuristics
            if any(s in name for s in tabular_substrings):
                category = ONLY_TAB
            elif any(s in name for s in image_substrings):
                category =  ONLY_IMG
            else:
                category = COMMON

        layer_classification[category].append(name)

        if debug:
            print(f"{name}: {category}")

    return layer_classification


def freeze_modality(model, modality=ONLY_TAB, debug=False):

    layers_classification = classify_parameters(model)

    for name, param in model.named_parameters():
        if name in layers_classification[modality]:
            param.requires_grad = False

        if debug:
            outcome = "FREEZE" if not param.requires_grad else "normal"
            print(f"- {name}: {outcome}")