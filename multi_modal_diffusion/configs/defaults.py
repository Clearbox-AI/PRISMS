import argparse
from diffusion_process.enums import DatasetType
from multi_modal_diffusion.custom_logger import DebugLogger

output_dir = "/mnt/storage/nacc_sub/tmp"
single_attn_active = True
cross_attn_active = True

# INITIALIZE CUSTOM DEBUG
debug_logger = DebugLogger(base_dir=output_dir)

def get_default_config():
    """
    Return a dictionary that merges:
      - Training defaults (like data_dir, batch_size, etc.)
      - Model defaults (like num_channels, channel_mult, etc.)
      - Diffusion defaults (like noise_schedule, etc.)

    So you have everything in a single config dictionary.
    """
    config = {}

    # 1. Basic training defaults
    config.update(
        dict(
            data_dir="",
            dataset_type=DatasetType.NACC_LATENTS,
            schedule_sampler="uniform",
            lr=1e-4,
            t_lr=1e-4,
            seed=42,
            weight_decay=0.0,
            lr_anneal_steps=0,
            batch_size=4,
            num_workers=0,
            microbatch=-1,
            ema_rate="0.9999",
            log_interval=10,
            devices=None,
            save_interval=300,
            output_dir=output_dir,
            resume_checkpoint="",
            use_fp16=False,
            fp16_scale_growth=1e-3,
            sample_fn="dpm_solver",
            class_cond=False,
            image_size="",      # can be overridden
            tabular_size="",    # can be overridden
            num_epochs=20,
        )
    )

    # 2. Model defaults
    config.update(
        dict(
            image_size="3,64,64",
            tabular_size="96",
            num_channels=192,
            num_res_blocks=1,  # e.g. was 2
            num_heads=2,
            num_heads_upsample=-1,
            num_head_channels=-1,
            cross_attention_resolutions="4,8,16",
            cross_attention_windows="1,1,1",
            cross_attention_shift=False,
            image_attention_resolutions="2,4,8,16",
            tabular_attention_resolutions="2,4,8,16",
            channel_mult="1,2,3,4",
            dropout=0.0,
            # This 'class_cond' was also in training defaults, so if you
            # want to unify them, you could remove from one or the other.
            # We'll keep the same name to keep consistent.
            # class_cond=False,
            use_checkpoint=False,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_fp16=False,
            image_type="2d",
            tabular_type="1d",
            freeze_mod=None,
            debug=True
        )
    )

    # 3. Diffusion defaults
    #    (You can pull these from your old `diffusion_defaults()` function)
    config.update(
        dict(
            learn_sigma=True,
            diffusion_steps=2000, #1000,
            noise_schedule="linear",
            timestep_respacing="",
            use_kl=False,
            predict_xstart=False,
            rescale_timesteps=True, #False,
            rescale_learned_sigmas=False,
        )
    )

    return config


def create_argparser():
    """
    Create an argparse parser from the merged config.
    """
    defaults = get_default_config()
    parser = argparse.ArgumentParser()
    for k, v in defaults.items():
        v_type = type(v)
        # If the default is None, parse it as string by default.
        if v is None:
            v_type = str
        # If it's a bool, use a custom boolean parser:
        elif isinstance(v, bool):
            v_type = str2bool
        elif k == "dataset_type":
            # Example of enumerating possible dataset types.
            parser.add_argument(
                f"--{k}", default=v, choices=[dt.value for dt in DatasetType]
            )
            continue

        parser.add_argument(f"--{k}", default=v, type=v_type)
    return parser


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def args_to_dict(args):
    """
    Optionally, if you want to convert argparse Namespace
    to a Python dict easily, you can use something like this.
    """
    return vars(args)
