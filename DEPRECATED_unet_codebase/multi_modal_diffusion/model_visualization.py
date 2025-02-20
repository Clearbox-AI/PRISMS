from model.mm_unet import MultimodalUNet
import torch
from torchviz import make_dot
from torch import nn


# Example dummy model initialization (adjust parameters as needed)
model = MultimodalUNet(
    image_size=(3, 64, 64),   # Example: (channels, height, width)
    tabular_size=10,          # Example: number of tabular features
    model_channels=64,
    image_out_channels=3,
    tabular_out_channels=10,
    num_res_blocks=2,
    cross_attention_resolutions={16, 8},
    image_attention_resolutions={16, 8},
    tabular_attention_resolutions={16, 8},
    dropout=0.1,
    channel_mult=(1, 2),
    use_checkpoint=False,
    use_fp16=False,
    num_heads=4,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    resblock_updown=True,
)

# Prepare dummy inputs
batch_size = 1
dummy_image = torch.randn(batch_size, 3, 64, 64)       # image input
dummy_tabular = torch.randn(batch_size, 10)            # tabular input
dummy_timesteps = torch.randint(low=0, high=1000, size=(batch_size,))
dummy_label = None  # If model is not class-conditional, keep this None

# Wrap the model in a simple forward so torchviz can trace it
class Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, tabular, timesteps, label):
        return self.model(image, tabular, timesteps, label)

wrapped_model = Wrapper(model)

# Run a forward pass to produce a graph
output_image, output_tabular = wrapped_model(dummy_image, dummy_tabular, dummy_timesteps, dummy_label)

# Create a visualization of the model graph
# Since the model outputs a tuple (image, tabular), pick one of them or combine them.
# Here, we'll just use the image output for graph visualization:
dot = make_dot(output_image, params=dict(wrapped_model.named_parameters()))

# Save the graph in PDF format
dot.render("multimodal_unet_graph", format="pdf")

print("Graph saved as multimodal_unet_graph.pdf")



