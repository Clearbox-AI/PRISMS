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
from multi_modal_diffusion.scripts.mm_training import load_training_data
from diffusion_process.multimodal_train_util import TrainLoop
from multi_modal_diffusion.resample import create_named_schedule_sampler

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

    logger.log("Creating data loader...")
    data_loader = load_training_data(args)

    # Infer image_size and tabular_size from data if not specified
    sample_batch = next(iter(data_loader))
    image_shape = sample_batch['image'].shape  # [batch_size, C, H, W]
    tabular_shape = sample_batch['tabular'].shape  # [batch_size, tabular_size]

    # Update args.image_size and args.tabular_size
    args.image_size = ','.join(map(str, image_shape[1:]))  # Exclude batch dimension
    args.tabular_size = str(tabular_shape[1])  # Exclude batch dimension

    # # Update args.image_size and args.tabular_size
    # args.image_size = '3,64,64'
    # args.tabular_size = '128' # Exclude batch dimension

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

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("Starting training...")

    num_epochs = 20
    eval_interval = 1
    num_eval_samples = 20

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data_loader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        num_epochs=num_epochs,
        lr=args.lr,
        t_lr=args.t_lr,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        class_cond=args.class_cond,
        sample_fn=args.sample_fn,
        eval_interval=eval_interval,
        num_eval_samples=num_eval_samples,
    ).run_loop()






if __name__ == "__main__":
    main()
