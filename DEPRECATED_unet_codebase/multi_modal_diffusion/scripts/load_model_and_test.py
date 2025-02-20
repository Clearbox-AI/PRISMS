import torch as th
import os
import argparse

from multi_modal_diffusion.common import set_seed_logger_random
from multi_modal_diffusion import dist_util, logger
from diffusion_process.multimodal_script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser
)
from multi_modal_diffusion.scripts.mm_training import create_argparser
from multi_modal_diffusion.architecture_utils.layers_initialization import reinitialize_specific_layers
from multi_modal_diffusion.architecture_utils.layers_classification import classify_parameters

RESTORE_MODEL_PATH = "/mnt/storage/lumir_three_stage_exp/stage_one"

def get_checkpoints(restore_path):

    subnames = ["ema", "model", "opt"]

    # Dictionary to hold the mappings
    latest_files = {
        "ema": None,
        "model": None,
        "opt": None
    }

    # Loop through each subname to find the latest file
    for subname in subnames:
        # List all files containing the subname
        matching_files = [f for f in os.listdir(restore_path) if subname in f]

        # Sort files by name (assumes that sorting by name corresponds to time ordering)
        matching_files.sort()

        # Take the last file if the list is not empty
        if matching_files:
            latest_files[subname] = os.path.join(restore_path, matching_files[-1])

    # Assign the paths to the variables
    return latest_files["ema"], latest_files["model"], latest_files["opt"]


def main():
    # Create argparser and parse arguments.
    # Make sure you use the same arguments as in training.
    args = create_argparser().parse_args()
    args = set_seed_logger_random(args)

    # Update args.image_size and args.tabular_size
    args.image_size = '3,64,64'
    args.tabular_size = '128' # Exclude batch dimension

    # Initialize logger (optional)
    logger.configure(args.output_dir)

    # Set up device
    dist_util.setup_dist(args.devices if args.devices else "cpu")

    # Create model and diffusion
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, [key for key in model_and_diffusion_defaults().keys()])
    )

    EMA_CKPT, MODEL_CKPT, _ = get_checkpoints(RESTORE_MODEL_PATH)

    # Load the model weights
    state_dict = th.load(MODEL_CKPT, map_location=dist_util.dev())
    model.load_state_dict(state_dict)

    # If you want to use the EMA weights (often preferred for inference):
    if os.path.exists(EMA_CKPT):
        ema_state = th.load(EMA_CKPT, map_location=dist_util.dev())
        model.load_state_dict(ema_state)


    ### INITIALIZE FOR PHASE 2 ###
    layer_classification = classify_parameters(model)
    common_layers = layer_classification["common"]
    reinitialize_specific_layers(model, common_layers)

    # Move model to device and set to eval mode
    model.to(dist_util.dev())
    model.eval()

    # Now let's create some dummy test input for inference
    batch_size = 4
    image_channels, image_height, image_width = [int(x) for x in args.image_size.split(',')]
    tabular_size = int(args.tabular_size)


    # generate samples:
    sample = diffusion.p_sample_loop(model, shape={"image": [batch_size, image_channels, image_height, image_width],
                                                  "tabular": [batch_size, tabular_size]},
                                    clip_denoised=True)
    generated_images = sample['image']
    generated_tabular = sample['tabular']

    import matplotlib.pyplot as plt
    import numpy as np
    img_tensor = generated_images[0].cpu()
    img_np = img_tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    plt.imshow(img_np)
    plt.show()

    print("Generated images shape:", generated_images.shape)
    print("Generated tabular shape:", generated_tabular.shape)

if __name__ == "__main__":
    main()
