def ntuple(n: int):
    """Converts input into an n-tuple."""
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse


def create_norm(norm_type: str, dim: int, eps: float = 1e-6) -> nn.Module:
    """
    Creates a normalization layer of the given type.
    Currently supports only "layernorm" or "np_layernorm".
    """
    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
    elif norm_type == "np_layernorm":
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
    else:
        raise ValueError(f'Unsupported norm type: {norm_type}')


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    AdaLN style shift & scale:
      out = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)