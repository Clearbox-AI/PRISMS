from pathlib import Path
import json
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

from data.loader import load_training_data
from utils.model import load_checkpoint
from utils.path_management import setup_exp_directory

from models.utils.model_loader import load_model
from enums.models.model_types import ModelType
from enums.training_versions import DiTTrainingVersion



@hydra.main(version_base=None, config_path="../../configs/experiments", config_name="vae_vs_dit_dist")
def main(cfg: DictConfig):

    # --------------------
    # 1) Build or load models
    # --------------------
    vae = load_model(model_type=ModelType.VAE)
    vae.requires_grad_(False)
    vae.eval()
    vae.to(device=cfg.execution_params.device)

    mm_diff_model = load_model(
        model_type=ModelType.DIFFUSION,
        model_variant=DiTTrainingVersion.base_dit_training
    )
    mm_diff_model.to(cfg.execution_params.device)

    # If the Diffusion model internally holds a DiT, extract it.
    dit_model = mm_diff_model.dit
    dit_model.to(cfg.execution_params.device)

    # --------------------
    # 2) Load checkpoint
    # --------------------
    checkpoint_path = cfg.execution_params.dit_checkpoint
    load_checkpoint(mm_diff_model, checkpoint_path, cfg.execution_params.device)
    mm_diff_model.eval()

    # --------------------
    # 3) Generate latents
    # --------------------
    num_batches = cfg.execution_params.num_batches
    all_generated_latents = []

    print("\nGenerating latents...")
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Generating", unit="batch"):
            latents, _ = mm_diff_model.sample(
                batch_size=cfg.execution_params.batch_size,
                table_data=torch.randn(cfg.execution_params.batch_size, 174, device=cfg.execution_params.device),
                cfg=1.0,
                steps=None,
                height=64,
                width=64,
                device=cfg.execution_params.device,
                save_path=None
            )
            all_generated_latents.append(latents)

    generated_latents = torch.cat(all_generated_latents, dim=0)  # (N, C, H, W)

    # --------------------
    # 4) Collect Original latents
    # --------------------
    all_original_latents = []
    train_loader = load_training_data(cfg)

    print("\nCollecting original latents...")
    for batch_idx, batch in tqdm(enumerate(train_loader), total=num_batches, desc="Collecting", unit="batch"):
        if batch_idx >= num_batches:
            break
        loaded_images = batch['image'].to(cfg.execution_params.device)
        with torch.no_grad():
            latents_batch = vae.encode(loaded_images.to(torch.float32))['latent_dist'].sample().data
            latents_batch = latents_batch * cfg.vae.scaling_factor
        all_original_latents.append(latents_batch)

    original_latents = torch.cat(all_original_latents, dim=0)

    # --------------------
    # 5) Compute Statistics
    # --------------------
    gen_stats = compute_latent_statistics(generated_latents)
    orig_stats = compute_latent_statistics(original_latents)

    # Print to console
    print("\n=== Latent Statistics Comparison ===")
    print(f"Generated Global Mean: {gen_stats['global_mean']:.4f}, Original Global Mean: {orig_stats['global_mean']:.4f}")
    print(f"Generated Global Var:  {gen_stats['global_var']:.4f}, Original Global Var:  {orig_stats['global_var']:.4f}")
    print("\nGenerated Per-Channel Means:", [round(x, 4) for x in gen_stats["mean_per_channel"]])
    print("Original Per-Channel Means:", [round(x, 4) for x in orig_stats["mean_per_channel"]])
    print("\nGenerated Per-Channel Variances:", [round(x, 4) for x in gen_stats["var_per_channel"]])
    print("Original Per-Channel Variances:", [round(x, 4) for x in orig_stats["var_per_channel"]])

    # --------------------
    # 6) Set up save directory and save results
    # --------------------
    save_dir = setup_exp_directory(
        base_path=cfg.execution_params.get("save_path") or Path(Path(__file__).resolve().parent.parent, "storage")
    )

    # Save statistics to text file
    save_statistics(file_path=str(Path(save_dir, "latent_statistics.json")), label="GENERATED.\n", stats=gen_stats)
    save_statistics(file_path=str(Path(save_dir, "latent_statistics.json")), label="ORIGINAL.\n", stats=orig_stats)

    # Plot histograms
    plt.figure(figsize=(12, 6))
    plt.hist(original_latents.flatten().cpu().numpy(), bins=200, alpha=0.5, density=True, label='Original')
    plt.hist(generated_latents.flatten().cpu().numpy(), bins=200, alpha=0.5, density=True, label='Generated')
    plt.title("Latent Value Distributions")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()

    hist_path = Path(save_dir, "latent_distributions.png")
    plt.savefig(hist_path)
    plt.close()
    print(f"Saved distribution comparison plot at {hist_path}")


def compute_latent_statistics(latents: torch.Tensor) -> dict:
    """
    Computes global mean, variance, per-channel mean, and per-channel variance.

    Args:
        latents (torch.Tensor): Latent tensor of shape (B, C, H, W).

    Returns:
        dict: A dictionary containing the computed statistics.
    """
    stats = {}
    stats["global_mean"] = torch.mean(latents).item()
    stats["global_var"] = torch.var(latents).item()
    stats["mean_per_channel"] = torch.mean(latents, dim=(0, 2, 3)).cpu().numpy().tolist()
    stats["var_per_channel"] = torch.var(latents, dim=(0, 2, 3)).cpu().numpy().tolist()
    return stats

def save_statistics(file_path: str, label: str, stats: dict):
    """
    Saves statistics to a JSON file. If the file exists, it loads the existing content,
    appends the new statistics, and writes everything back.

    Args:
        file_path (str): Path to the JSON file.
        label (str): A string label (e.g., "GENERATED", "ORIGINAL").
        stats (dict): The statistics dictionary to be saved.
    """
    file_path = Path(file_path)

    # Load existing data if file exists
    if file_path.exists():
        try:
            with open(file_path, 'r') as f:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):  # Ensure it's a list
                    existing_data = []
        except json.JSONDecodeError:
            existing_data = []
    else:
        existing_data = []

    # Append the new statistics entry
    existing_data.append({"type": label, "statistics": stats})

    # Save everything back to the file
    with open(file_path, 'w') as f:
        json.dump(existing_data, f, indent=4)

    print(f"Saved {label} statistics at {file_path}")



if __name__ == "__main__":
    main()
