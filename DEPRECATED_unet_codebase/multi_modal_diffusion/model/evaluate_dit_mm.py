import torch
import numpy as np
import itertools
from scipy.spatial.distance import cdist
import ot  # from POT (Python Optimal Transport)
from unittest.mock import MagicMock
from diffusion_process.dataloaders import load_training_data
from diffusion_process.enums import DatasetType
from multi_modal_diffusion.model.train_dit_mm import MultiModalDiffusion
from multi_modal_diffusion.model.dit_mm import MultiModalDiT
import numpy as np
from torch import optim
from tqdm import tqdm


def load_diffusion_checkpoint(
        checkpoint_path: str,
        model: MultiModalDiffusion,
        device: str = "cuda",
        load_optimizer: bool = False
):
    """
    Load a saved checkpoint into the MultiModalDiffusion model.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        model: Initialized MultiModalDiffusion model (architecture must match)
        device: Target device for loading
        load_optimizer: Whether to also return optimizer state

    Returns:
        Dictionary containing:
        - model: Loaded model
        - optimizer (optional): Optimizer state
        - training_info: step, epoch, loss
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 1. Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # 2. Prepare return dict
    result = {
        'model': model,
        'training_info': {
            'step': checkpoint['step'],
            'epoch': checkpoint['epoch'],
            'loss': checkpoint['loss']
        }
    }

    # 3. Load optimizer if requested
    if load_optimizer:
        optimizer = optim.AdamW(model.parameters(), lr=0)  # lr will be overwritten
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        result['optimizer'] = optimizer

    return result

# ------------------------------------------------------------------
# 1) Utilities: MMD + Wasserstein
# ------------------------------------------------------------------
def rbf_kernel(x, y, gamma=None):
    """
    Computes the RBF kernel between two sets of vectors x and y.
    x: (N, D), y: (M, D)
    """
    x = x.unsqueeze(1)  # -> (N, 1, D)
    y = y.unsqueeze(0)  # -> (1, M, D)
    diff = x - y        # -> (N, M, D)
    dist_sq = diff.pow(2).sum(dim=2)  # (N, M)

    # Median heuristic for gamma if not provided
    if gamma is None:
        median_sq = torch.median(dist_sq)
        if median_sq.item() == 0:
            median_sq = torch.tensor(1.0, device=dist_sq.device)
        gamma = 1.0 / (2.0 * median_sq)

    return torch.exp(-gamma * dist_sq)


def mmd_rbf(x, y, gamma=None):
    """
    Computes Maximum Mean Discrepancy (MMD) between x & y using RBF kernel.
    x: (N, D), y: (M, D)
    Returns a single scalar (tensor).
    """
    xx = rbf_kernel(x, x, gamma=gamma)
    yy = rbf_kernel(y, y, gamma=gamma)
    xy = rbf_kernel(x, y, gamma=gamma)
    return xx.mean() + yy.mean() - 2.0 * xy.mean()


def multi_dim_wasserstein_distance(real_data, fake_data):
    """
    Computes the 1-Wasserstein distance (EMD) using optimal transport (POT).
    real_data: (N, D), fake_data: (M, D) - Tensors or ndarrays.
    """
    if isinstance(real_data, torch.Tensor):
        real_data = real_data.detach().cpu().numpy()
    if isinstance(fake_data, torch.Tensor):
        fake_data = fake_data.detach().cpu().numpy()

    n = real_data.shape[0]
    m = fake_data.shape[0]
    a = np.ones((n,)) / n  # uniform distribution
    b = np.ones((m,)) / m

    cost_matrix = cdist(real_data, fake_data, metric='euclidean')
    wdist = ot.emd2(a, b, cost_matrix)
    return wdist


# ------------------------------------------------------------------
# 2) Main Evaluation Function
# ------------------------------------------------------------------
def evaluate_tabular_distribution(
    model,
    data_loader,
    device='cuda',
    max_batches=10,
    compute_baseline=False,
    baseline_repeats=5,
    baseline_seed=123
):
    """
    Evaluates how closely model-generated tabular data matches real tabular data,
    by iterating through 'max_batches' from data_loader. Also (optionally) computes
    baseline metrics (Real-vs-Real) using repeated splits.

    Args:
        model (nn.Module): Your trained diffusion model with .sample(...) method.
        data_loader (DataLoader): Yields batches like {'image': ..., 'tabular': ...}.
        device (str): 'cpu' or 'cuda'.
        max_batches (int): Number of batches to evaluate for real-fake metrics.
        compute_baseline (bool): If True, compute Real-vs-Real baseline metrics.
        baseline_repeats (int): How many times to do the Real-vs-Real split & average.
        baseline_seed (int): Seed for reproducible splits.

    Returns:
        wdist (float): Wasserstein distance (Real vs. Generated)
        mmd_val (float): MMD (Real vs. Generated)
        baseline_wdist_avg (float or None): Average WD over repeated Real-vs-Real splits
        baseline_mmd_avg (float or None): Average MMD over repeated Real-vs-Real splits
    """
    model.eval()

    real_tab_batches = []
    fake_uncond_batches = []
    fake_cond_batches = []

    # --------------------------------------------------
    # A) Gather Real and Generated Data (Iterative)
    # --------------------------------------------------
    num_batches = len(data_loader) if max_batches is None else max_batches
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(itertools.islice(data_loader, num_batches),
                                               total=num_batches,
                                               desc="Processing Batches",
                                               unit="batch")):
            real_tab = batch['tabular'].to(device)
            # Generate Unconditioned
            _, tab_uncond  = model.sample(
                batch_size=real_tab.shape[0],
                table_data=None,
                cfg=1.0,
                steps=None,
                height=64,
                width=64,
                device=device
            )

            # Generate Conditioned
            _, tab_cond = model.sample(
                batch_size=real_tab.shape[0],
                table_data=real_tab,
                cfg=1.0,
                steps=None,
                height=64,
                width=64,
                device=device
            )

            real_tab_batches.append(real_tab)
            fake_uncond_batches.append(tab_uncond)
            fake_cond_batches.append(tab_cond)

    if len(real_tab_batches) == 0:
        # In case data_loader is empty or max_batches=0
        return float('nan'), float('nan'), None, None

    # Concatenate all
    real_tab_all = torch.cat(real_tab_batches, dim=0).float()  # (N, D)
    fake_uncond_all = torch.cat(fake_uncond_batches, dim=0).float().to(device)  # (N, D)
    fake_cond_all = torch.cat(fake_cond_batches, dim=0).float().to(device)  # (N, D)

    # --------------------------------------------------
    # B) Compute Real-vs-Generated Metrics
    # --------------------------------------------------
    mmd_uncond = mmd_rbf(real_tab_all, fake_uncond_all)
    wdist_uncond = multi_dim_wasserstein_distance(real_tab_all, fake_uncond_all)

    mmd_cond = mmd_rbf(real_tab_all, fake_cond_all)
    wdist_cond = multi_dim_wasserstein_distance(real_tab_all, fake_cond_all)

    # Convert to float or .item()
    mmd_uncond = mmd_uncond.item()
    mmd_cond = mmd_cond.item()
    # wdist is already float from multi_dim_wasserstein_distance

    # --------------------------------------------------
    # C) (Optional) Compute Baseline Real-vs-Real
    # --------------------------------------------------
    baseline_wdist_avg = None
    baseline_mmd_avg = None

    if compute_baseline:
        # We'll do multiple splits, each with a different seed
        wdist_list = []
        mmd_list = []

        n_samples = real_tab_all.shape[0]
        for i in range(baseline_repeats):
            # Fix the seed for reproducible splits
            torch.manual_seed(baseline_seed + i)

            # Shuffle indices
            indices = torch.randperm(n_samples, device=device)
            split_idx = n_samples // 2  # e.g. 50/50 split

            subset1 = real_tab_all[indices[:split_idx]]
            subset2 = real_tab_all[indices[split_idx:]]

            # Compute WD + MMD for this split
            wdist_i = multi_dim_wasserstein_distance(subset1, subset2)
            mmd_i = mmd_rbf(subset1, subset2).item()

            wdist_list.append(wdist_i)
            mmd_list.append(mmd_i)

        # Average results over baseline_repeats
        baseline_wdist_avg = float(np.mean(wdist_list))
        baseline_mmd_avg = float(np.mean(mmd_list))

    # --------------------------------------------------
    # Return the metrics
    # --------------------------------------------------
    return (wdist_uncond, mmd_uncond, wdist_cond, mmd_cond, baseline_wdist_avg, baseline_mmd_avg)


# ------------------------------------------------------------------
# 3) Example Usage in main()
# ------------------------------------------------------------------
def main():

    # ------------------------------------------------
    # Initialize your DataLoader
    # ------------------------------------------------
    mock_args = MagicMock()
    mock_args.dataset_type = DatasetType.IMAGE_TABULAR
    mock_args.data_dir = "/mnt/dataset_storage/data/nacc_dataset/nacc_subset/middle_slice"
    mock_args.batch_size = 8
    mock_args.num_workers = 0
    train_loader = load_training_data(mock_args)

    # ------------------------------------------------
    # Build your model (architecture as needed)
    # ------------------------------------------------
    qkv_ratio = [0.5, 1.0]
    mlp_ratio = [0.5, 4.0]
    depth = 16

    dit_model = MultiModalDiT(
        input_size=64,
        patch_size=4,
        in_channels=3,
        dim=256,
        depth=depth,
        head_dim=32,
        multiple_of=64,
        qkv_multipliers=np.linspace(qkv_ratio[0], qkv_ratio[1], num=depth, dtype=float),
        ffn_multipliers=np.linspace(mlp_ratio[0], mlp_ratio[1], num=depth, dtype=float),
        use_patch_mixer=True,
        patch_mixer_depth=4,
        patch_mixer_dim=512,
        patch_mixer_qkv_ratio=1.0,
        patch_mixer_mlp_ratio=4.0,
        use_bias=False,
        num_experts=8,
        expert_capacity=2.0,
        experts_every_n=2,
        num_tab_columns=174,
        tab_groups=10,
        out_table_features=174
    )

    mm_diff_model = MultiModalDiffusion(
        dit=dit_model,
        vae=None,  # if pixel-space diffusion
        sigma_min=0.002,
        sigma_max=80,
        p_mean=-0.6,
        p_std=1.2,
        sigma_data=0.9,
        num_steps=18
    )

    # ------------------------------------------------
    # Load model checkpoint
    # ------------------------------------------------
    checkpoint_path = "/mnt/storage/nacc_sub/mm_dit_NO_LAT/models_20_uncod/checkpoint_step_24200_final.pt"
    loaded = load_diffusion_checkpoint(
        checkpoint_path=checkpoint_path,
        model=mm_diff_model,
        device="cuda",
        load_optimizer=True
    )

    # Get the loaded model
    mm_diff_model = loaded['model']

    # ------------------------------------------------
    # Evaluate distribution
    # ------------------------------------------------

    baseline_repeats = 10
    baseline_seed = 123
    (wdist_uncond, mmd_uncond, wdist_cond, mmd_cond, baseline_wdist, baseline_mmd) = evaluate_tabular_distribution(
        model=mm_diff_model,
        data_loader=train_loader,
        device="cuda",
        max_batches=None,
        compute_baseline=True,
        baseline_repeats=baseline_repeats,
        baseline_seed=baseline_seed
    )

    if baseline_wdist is not None:
        print(f"[BASELINE] over {baseline_repeats} splits (seed={baseline_seed})")
        print(f"Avg. Wasserstein Distance: {baseline_wdist:.4f}")
        print(f"Avg. MMD (RBF): {baseline_mmd:.4f}")

    print(f"\n[REAL vs UNCONDITIONED GENERATED]")
    print(f"Wasserstein Distance: {wdist_uncond:.4f}")
    print(f"MMD (RBF): {mmd_uncond:.4f}")

    print(f"\n[REAL vs CONDITIONED GENERATED]")
    print(f"Wasserstein Distance: {wdist_cond:.4f}")
    print(f"MMD (RBF): {mmd_cond:.4f}")



if __name__ == "__main__":
    main()
