import os
import torch as th
from multi_modal_diffusion import dist_util, logger
from multi_modal_diffusion.resample import create_named_schedule_sampler
from diffusion_process.dataloaders import load_training_data  # Not used in inference
from multi_modal_diffusion.model.model_setup import create_model_and_diffusion
from multi_modal_diffusion.configs.defaults import get_default_config
from multi_modal_diffusion.training.pipeline import TrainingStep  # Not used in inference
from multi_modal_diffusion.common import set_seed_logger_random
from multi_modal_diffusion.sampler import DPM_Solver  # Adjust import path as necessary
import matplotlib.pyplot as plt


class BaseInferenceStep:
    def __init__(self):
        """
        Initialize the inference step with placeholders for model, diffusion, and solver.
        """
        self.model = None
        self.diffusion = None
        self.schedule_sampler = None
        self.device = None
        self.dpm_solver = None
        self.alphas_cumprod = None
        self.ema_models = []
        self.optimizer_state = None

    def setup(self, args, checkpoint_step, checkpoint_dir):
        """
        Set up the inference environment, load the model and diffusion parameters from checkpoint.

        Args:
            args: Namespace or dictionary containing configuration parameters.
            checkpoint_step (int): The step number of the checkpoint to load.
            checkpoint_dir (str): Directory where checkpoints are saved.
        """
        # 1. Set seed and configure logger
        args = set_seed_logger_random(args)
        logger.configure(args.output_dir)

        # 2. Distributed setup
        dist_util.setup_dist(args.devices if hasattr(args, 'devices') else "cpu")
        self.device = dist_util.dev()

        # 3. Load model, EMA models, optimizer state, and diffusion parameters from checkpoint
        self.model, self.ema_models, self.optimizer_state, self.alphas_cumprod = self.load_checkpoint(
            checkpoint_dir, checkpoint_step, self.device, args
        )

        # 4. Initialize DPM_Solver with loaded alphas_cumprod
        self.dpm_solver = DPM_Solver(
            model=self.model,
            alphas_cumprod=self.alphas_cumprod,
            predict_x0=False,  # Use noise prediction mode
            thresholding=False,  # Disable dynamic thresholding
            guidance_type="uncond",  # Adjust if using guidance (e.g., "classifier-free")
            max_val=1.0,  # Only relevant if thresholding is enabled
            model_kwargs={},  # Additional kwargs if needed
            rescale=False  # Typically False for latent diffusion
        )

    def load_checkpoint(self, checkpoint_dir, step, device, args):
        """
        Load model, EMA models, optimizer, and diffusion parameters from checkpoint.

        Args:
            checkpoint_dir (str): Directory where checkpoints are saved.
            step (int): The step number of the checkpoint to load.
            device (torch.device): The device to map tensors to.
            args: Namespace or dictionary containing configuration parameters.

        Returns:
            model (torch.nn.Module): Loaded model.
            ema_models (list of torch.nn.Module): Loaded EMA models.
            optimizer_state (dict): Loaded optimizer state.
            alphas_cumprod (torch.Tensor): Loaded alphas_cumprod tensor.
        """
        # 1. Load main model
        model_path = os.path.join(checkpoint_dir, f"model{step:06d}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        model_state = th.load(model_path, map_location=device)

        # Create model and diffusion objects (without initializing them with training parameters)
        model, diffusion = create_model_and_diffusion(
            **{k: getattr(args, k) for k in get_default_config().keys()}
        )

        # Load state dict
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()

        # 2. Load EMA models
        ema_models = []
        for rate in args.ema_rate:
            ema_path = os.path.join(checkpoint_dir, f"ema_{rate}_{step:06d}.pt")
            if not os.path.exists(ema_path):
                logger.warn(f"EMA checkpoint not found at {ema_path}, skipping.")
                continue
            ema_state = th.load(ema_path, map_location=device)
            ema_model, _ = create_model_and_diffusion(
                **{k: getattr(args, k) for k in get_default_config().keys()}
            )
            ema_model.load_state_dict(ema_state)
            ema_model.to(device)
            ema_model.eval()
            ema_models.append(ema_model)

        # 3. Load optimizer state (optional for inference)
        optimizer_state = None
        optimizer_path = os.path.join(checkpoint_dir, f"opt{step:06d}.pt")
        if os.path.exists(optimizer_path):
            optimizer_state = th.load(optimizer_path, map_location=device)
            logger.info(f"Loaded optimizer state from {optimizer_path}")
        else:
            logger.warn(f"Optimizer checkpoint not found at {optimizer_path}, skipping.")

        # 4. Load diffusion parameters
        diffusion_path = os.path.join(checkpoint_dir, f"diffusion{step:06d}.pt")
        if not os.path.exists(diffusion_path):
            raise FileNotFoundError(f"Diffusion checkpoint not found at {diffusion_path}")
        diffusion_state = th.load(diffusion_path, map_location=device)
        alphas_cumprod = diffusion_state['alphas_cumprod'].to(device)

        logger.info(f"Loaded model, EMA models, and diffusion parameters from step {step}")

        return model, ema_models, optimizer_state, alphas_cumprod

    def sample_latents(self, batch_size, latent_shape, tabular_size, steps=20):
        """
        Generate sampled latents using the DPM_Solver.

        Args:
            batch_size (int): Number of samples to generate.
            latent_shape (tuple): Shape of the latent space (C, H, W).
            tabular_size (int): Dimension of the tabular data.
            steps (int): Number of sampling steps.

        Returns:
            sampled (dict): Dictionary containing sampled latents.
        """
        # Initialize latent noise (4 channels)
        latent_channels, latent_height, latent_width = latent_shape
        x_T = {
            "image": th.randn([batch_size, latent_channels, latent_height, latent_width], device=self.device),
            "tabular": th.randn([batch_size, tabular_size], device=self.device)
        }

        # Perform sampling
        with th.no_grad():
            sampled = self.dpm_solver.sample(
                x=x_T,
                steps=steps,
                t_start=None,  # Use default t_start (self.diffusion.T)
                t_end=None,  # Use default t_end (1 / total_N)
                order=2,  # Solver order; typically 2 or 3
                skip_type="logSNR",  # Choose based on your noise schedule ('logSNR', 'time_uniform', etc.)
                method="singlestep",  # 'singlestep' is recommended for fixed-step solvers
                denoise=False,  # Disable denoising at the final step if not needed
                solver_type="dpm_solver",  # 'dpm_solver' is generally preferred over 'taylor'
                atol=0.0078,  # Absolute tolerance for adaptive (irrelevant for singlestep)
                rtol=0.05  # Relative tolerance for adaptive (irrelevant for singlestep)
            )
        return sampled

    def visualize_latent_channels(self, sampled, sample_index=0):
        """
        Visualize individual latent channels for a specific sample.

        Args:
            sampled (dict): Dictionary containing sampled latents.
            sample_index (int): Index of the sample to visualize.
        """
        sampled_latents = sampled["image"]  # Shape: [batch_size, 4, H, W]
        num_channels = sampled_latents.shape[1]

        fig, axs = plt.subplots(1, num_channels, figsize=(15, 5))
        for c in range(num_channels):
            channel = sampled_latents[sample_index, c].cpu().numpy()
            axs[c].imshow(channel, cmap='gray')
            axs[c].axis('off')
            axs[c].set_title(f'Latent Channel {c + 1}')
        plt.show()

    def run(self, batch_size, latent_shape, tabular_size, steps=20, visualize=False, sample_index=0):
        """
        Execute the inference pipeline: sample latents and optionally visualize them.

        Args:
            batch_size (int): Number of samples to generate.
            latent_shape (tuple): Shape of the latent space (C, H, W).
            tabular_size (int): Dimension of the tabular data.
            steps (int): Number of sampling steps.
            visualize (bool): Whether to visualize the sampled latents.
            sample_index (int): Index of the sample to visualize if `visualize=True`.

        Returns:
            sampled (dict): Dictionary containing sampled latents.
        """
        sampled = self.sample_latents(batch_size, latent_shape, tabular_size, steps)

        if visualize:
            self.visualize_latent_channels(sampled, sample_index=sample_index)

        return sampled
