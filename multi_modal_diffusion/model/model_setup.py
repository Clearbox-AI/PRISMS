"""
This code is extended from guided_diffusion: https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/scripts_util.py
"""

from multi_modal_diffusion.model.mm_unet import MultimodalUNet
from multi_modal_diffusion.architecture_utils.layers_classification import freeze_modality
from einops import rearrange
from diffusion_process import multimodal_gaussian_diffusion as gd
from diffusion_process.multimodal_respace import SpacedDiffusion, space_timesteps

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
        freeze_mod=None,
        debug=False,
        **kwargs
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
        freeze_mod=freeze_mod,
        debug=debug,
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
        freeze_mod=None,
        debug=False,
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
        image_out_channels=(4 if not learn_sigma else 8), #(3 if not learn_sigma else 6),
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
        resblock_updown=resblock_updown,
        debug=debug
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
