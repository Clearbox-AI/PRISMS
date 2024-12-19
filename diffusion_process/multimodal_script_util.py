"""
This code is extended from guided_diffusion: https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/scripts_util.py
"""

import argparse
from einops import rearrange
from diffusion_process import multimodal_gaussian_diffusion as gd
from diffusion_process.multimodal_respace import SpacedDiffusion, space_timesteps
from multi_modal_diffusion.model.mm_unet import MultimodalUNet
import torch as th
from multi_modal_diffusion.architecture_utils.layers_classification import freeze_modality


def diffusion_defaults():
    """
    Defaults for multi-modal training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )


def model_defaults():
    """
    Defaults for multi-modal training.
    """
    res = dict(
        image_size="3,64,64",  # Changed from "16,3,64,64" to "3,64,64"
        tabular_size="96",
        num_channels=192,
        num_res_blocks=1, #2
        num_heads=2, #2
        num_heads_upsample=-1,
        num_head_channels=-1,
        cross_attention_resolutions="4,8,16", # 2,4,8
        cross_attention_windows="1,1,1", # 1,4,8
        cross_attention_shift=False, # True
        image_attention_resolutions="2,4,8,16", # 2,4,8
        tabular_attention_resolutions="2,4,8,16", # -1
        channel_mult="1,2,3,4", # ""
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True, # False
        use_fp16=False,
        image_type="2d",  # Changed from "2d+1d" to "2d"
        tabular_type="1d",
        freeze_mod="only_image" # here put what you want to freeze
    )
    return res


def model_and_diffusion_defaults():
    res = model_defaults()
    res.update(diffusion_defaults())
    return res


def create_model_and_diffusion(
        image_size,
        tabular_size,
        learn_sigma,
        num_channels,
        num_res_blocks,
        channel_mult,
        num_heads,
        num_head_channels,
        num_heads_upsample,
        cross_attention_resolutions,
        cross_attention_windows,
        cross_attention_shift,
        image_attention_resolutions,
        tabular_attention_resolutions,
        dropout,
        diffusion_steps,
        noise_schedule,
        timestep_respacing,
        use_kl,
        predict_xstart,
        rescale_timesteps,
        rescale_learned_sigmas,
        use_checkpoint,
        use_scale_shift_norm,
        resblock_updown,
        use_fp16,
        image_type="2d",
        tabular_type="1d",
        class_cond=False,
        freeze_mod=None
):
    model = create_model(
        image_size=image_size,
        tabular_size=tabular_size,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        cross_attention_resolutions=cross_attention_resolutions,
        image_attention_resolutions=image_attention_resolutions,
        tabular_attention_resolutions=tabular_attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        freeze_mod=freeze_mod
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def create_model(
        image_size,
        tabular_size,
        num_channels,
        num_res_blocks,
        channel_mult="",
        learn_sigma=False,
        class_cond=False,
        use_checkpoint=False,
        cross_attention_resolutions="2,4,8",
        image_attention_resolutions="2,4,8",
        tabular_attention_resolutions="2,4,8",
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        dropout=0,
        use_fp16=False,
        resblock_updown=True,
        freeze_mod=None
):
    # Parse sizes
    image_size = tuple(int(x) for x in image_size.split(','))
    tabular_size = int(tabular_size)

    # Adjust channel_mult based on image_size
    if channel_mult == "":
        if image_size[-1] == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size[-1] == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size[-1] == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size[-1] == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size[-1]}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    cross_attention_resolutions = [int(i) for i in cross_attention_resolutions.split(',')]
    image_attention_resolutions = [int(i) for i in image_attention_resolutions.split(',')]
    tabular_attention_resolutions = [int(i) for i in tabular_attention_resolutions.split(',')]

    unet_model = MultimodalUNet(
        image_size=image_size,
        tabular_size=tabular_size,
        model_channels=num_channels,
        image_out_channels=(3 if not learn_sigma else 6),
        tabular_out_channels=(tabular_size if not learn_sigma else tabular_size * 2),
        num_res_blocks=num_res_blocks,
        cross_attention_resolutions=cross_attention_resolutions,
        image_attention_resolutions=image_attention_resolutions,
        tabular_attention_resolutions=tabular_attention_resolutions,
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=None,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown
    )

    if freeze_mod:
        freeze_modality(unet_model, modality=freeze_mod, debug=True)

    return unet_model


def create_gaussian_diffusion(
        *,
        steps=1000,
        learn_sigma=False,
        sigma_small=False,
        noise_schedule="linear",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    defaults = model_and_diffusion_defaults()
    add_dict_to_argparser(parser, defaults)
    args = parser.parse_args()

    # Convert args to a dictionary
    args_dict = args_to_dict(args, list(defaults.keys()))

    # Create model and diffusion
    model, diffusion = create_model_and_diffusion(**args_dict)

    # Print model and diffusion to verify creation
    print("Model created:")
    print(model)
    print("\nDiffusion process created:")
    print(diffusion)

    # Example usage:
    # Create dummy input data
    batch_size = 4
    image_channels, image_height, image_width = map(int, args.image_size.split(','))
    tabular_size = int(args.tabular_size)

    # Create random tensors as dummy inputs
    image_input = th.randn(batch_size, image_channels, image_height, image_width)
    tabular_input = th.randn(batch_size, tabular_size)

    # Dummy timesteps
    t = th.randint(low=0, high=diffusion.num_timesteps, size=(batch_size,))

    # Run the model
    model_output = model(image_input, tabular_input, t)

    print("\nModel output:")
    print(model_output)
