# Introduction

**PRISMS** is a **multimodal generation** project funded under HEAL ITALIA’s cascade call, spoke 2 ([https://www.healitalia.eu/bac/](https://www.healitalia.eu/bac/)). Our implementatio aims to jointly create **brain T1 images** and the corresponding **tabular** morphometric/demographic **data**, while preserving **clinical coherence**, **downstream utility**, and **privacy constraints**.

To address the scarcity of truly multimodal datasets, we partially integrated the **NACC** and **ADNI** cohorts, harmonizing diagnoses ([https://aiformedresearch.github.io/aiformedresearch/](https://aiformedresearch.github.io/aiformedresearch/)), tabular schema, and preprocessing pipeline, in order to obtain a **comparable subsample** with individual-level traceability.

On the modeling side, PRISMS adopts **latent-space diffusion** with a **DiT** (Vision Transformer) under an **EDM** scheme, combined with an **image VAE** and a **TabSyn-style tabular VAE** (token-per-column). The two modalities interact via **symmetric cross-attention** with **adaptive gating**, while training employs **asymmetric schedulings**, **GradNorm**, **soft TTUR**, **self-conditioning**, and **EMA** to balance branch speed and stability. The goal is to generate **consistent image-tab pairs**, with plausible anatomical quality and useful tabular statistics, despite the **low-sample regime** (~1000 subjects).

For **privacy** and multi-site collaboration, the project was made **swarm-compatible** using **HPE Swarm Learning**, avoiding data centralization (only weights/updates are exchanged). Evaluation combines metrics on **tabular** (SURE), **images** (SSIM/MS-SSIM, FID with Synthetic_Images_Metrics_Toolkit), **multimodal coherence**, and **privacy** (DCR, MIA, clone detection). Empirical results show **good image reconstructions**, while the tabular branch requires further **tuning** to mitigate **skew**, **under-dispersion**, and **weak correlations**.

# Dataset

## Dataset formation (NACC + ADNI)

The dataset integrates two cohorts — **NACC** and **ADNI** — harmonized to ensure clinical and technical comparability. **NACC diagnoses** were **recomputed according to ADNI criteria** (binary setting: cognitively normal *CN* vs. Alzheimer’s disease *AD*), while ADNI retains its native criteria; this yields a more conservative but cross-study coherent subset. We include 2D **T1-weighted brain MRI only**, selecting the **first eligible visit** per subject. **ADNI tabular metadata** were mapped onto the **NACC schema** (shared nomenclature; unmatched variables marked missing or recovered via targeted queries). The procedure involved: label harmonization, schema unification, de-duplication by subject/visit keys, and an **individual-level merge** with traceability of source and filters.

### Rationale for merging

- **Scarcity of truly multimodal datasets** (image + tabular measures for the same patient), a crucial need for the project.
- **Potential increase in statistical power** and more stable estimates under aligned criteria.
- **Alignment to the ADNI standard**, facilitating comparisons and benchmarks against the literature.
- **Generalization/robustness**: assessment of *domain shift* across sites/scanners and improved external validity.

## Dataset description — tabular part

The dataset comprises **164 features** derived from structural MRI (T1). It includes two demographic variables (sex, age) and a broad morphometric set: **global volumes** (intracranial, brain, GM, WM, CSF, WMH), **ventricular volumes** (lateral and third), **hippocampus** (total and lateralized), and **lobar metrics** of the cortical gray matter (frontal, parietal, temporal, occipital; left, right, and total). The core component is cortical parcellation (Desikan–Killiany-like schema) with **GM volumes** and **mean cortical thicknesses** for each region, reported **per hemisphere**. Measures combine **lateralization** and **aggregates** (total/lobar), providing a fine-grained representation of brain morphology.

## NACC/ADNI Dataset Pipeline

The dataset originates from a **subsample of patients present in the NACC** and ADNI datasets (n≈1392), for whom we assembled **brain MRIs** and the associated **anatomical measures** in tabular form. The following focuses on the steps taken in the image preprocessing and standardization pipeline.

## How we prepared the images

1. **Selecting the best exam per patient**
    
    For each subject we chose the latest available exam to reflect the **most up-to-date clinical state**. In parallel, we paired each image with the **anatomical measures**.
    
2. **Orientation standardization**
    
    MRIs can be stored with different axis conventions (left/right, anterior/posterior). We converted all images to the same convention (RAS) so that **left and right** are consistent and each structure appears **in the same orientation**.
    
3. **Brain extraction (skull stripping)**
    
    Non-brain parts (bone, skin, paranasal sinuses) can confound analyses and measurements. Starting from a **reference brain mask** (in MNI space), we adapted it to each subject and applied it to **retain only brain tissue**. This ensures evaluations focus on parenchyma and CSF spaces.
    
4. **Bias-field correction**
    
    MRIs can exhibit artificially brighter or darker zones due to the electromagnetic field. A standard correction (N4) reduces these inhomogeneities, yielding **more uniform contrast** between gray matter, white matter, and CSF.
    
5. **Alignment to a reference brain**
    
    To compare different people, one must “speak the same anatomical language.” Each image was **aligned** to a **standard template (MNI152, 1 mm)** using translation, rotation, and scaling. Thus, major structures (ventricles, lobes, cerebellum) **fall in the same locations** across subjects, facilitating comparisons and group averages.
    
6. **Size standardization**
    
    All volumes were resampled to the **same size (256×256×256 voxels)**, preserving the image’s **anatomical center**. This makes extracting the **central axial slice** simpler and consistent.
    
7. **Extraction of the central section**
    
    From the normalized 3D image we extract the **middle axial slice**. This section crosses key areas (corpus callosum, basal ganglia, lateral ventricles) and serves as a **visual summary** of the patient’s brain anatomy in a 2D format suitable for models and tables.
    
8. Finally, we performed image “quality” checks:
    1. Visual verification that the **mask** covers the brain without “eating” useful tissue.
    2. **Contrast** check pre/post correction to avoid spurious overly bright/dark regions.
    3. Quick comparison with the **template**: major structures must plausibly overlap.

### Why these transformations are needed

- **Clinical comparability**: by removing—or at least reducing—technical variability and non-brain parts, observed differences between patients more faithfully reflect **biological features** and **disease status**.
- **Measurement reliability**: orientation uniformity, bias correction, and alignment to the same anatomical space improve the **stability** of volumes and cortical thicknesses.
- **Reproducibility**: the procedure is identical for all patients; this enables **replication** of results and integration of new subjects under the same standard.

# Latent Space

## Choice of the **latent space — images**

### Initial hypothesis

Given limited data, training an autoencoder *from scratch* was excluded; we focused on **pretrained VAEs** that are robust and established in the *latent diffusion* ecosystem. The goal was to compress T1 scans into a **low-dimensional latent** to simplify the diffusion model, stabilize optimization, and contain compute costs.

## Experiments performed

- **Screening public VAEs** (the *Stable Diffusion* and Monai families): we tested open-source variants (KL-f8 and derivatives), evaluating them on T1 reconstructions after encode/decode and on downstream training stability. The **Stability AI** line,widely used as a *building block* for *latent diffusion,* showed the best compromise among reconstruction quality, speed, and implementation/tooling maturity, despite Monai models being trained on medical data.
- **Qualitative/quantitative evaluations**: SSIM/PSNR on reconstructions and visual anatomical checks. In early iterations, Stability AI models reconstructed better, likely thanks to their massive training data, despite the intuitive expectation that Monai models might perform better due to domain proximity.
- **Final decision**: adoption of a **pretrained Stability AI VAE** as the latent *encoder/decoder* for the image branch.

## Choice of the latent space — tabular

### Initial hypothesis

In the initial experimental phase we prioritized the harder problem, image generation, so **no tabular VAE** was sought at first: for the tabular branch we worked **directly in input space** (diffusion on normalized features), deferring latent encoding of tabulars to later phases.

### Experiments performed

We later adopted a **“TabSyn” token-per-column VAE**: categoricals are modeled as **logits**, numerics with an **MDN decoder** that aims to capture tails and multimodality; encoder/decoder are **Transformers** over CLS-token sequences. As per the paper, we used **β-anneal** and **free-bits** to avoid KL collapse and, in diffusion training, we pass the **posterior μ** (deterministic latent), with a **temperature τ** ramping up to increase explorability.

The classic VAE idea applies: obtain **regular, comparable tokens** to better preserve statistical structure versus input space.

# Diffusion Model Architecture

## Generator **core architecture**

### Phase 1 — **Multimodal UNet** + DPM-solver (unsatisfactory results)

- **Initial choice**: start from a UNet, the historically prevalent *backbone* in diffusion models, extended to the **multimodal** context (image + tabular), initially without label conditioning, and **DPM-Solver** to accelerate *sampling*.
- **Implementation**: adaptation of one of the few existing multimodal *codebases*, adjusted to our project context with a different multimodality type.
- **Difficulties encountered**:
    - **Fragility to architectural changes** due to the complexity of moving from one multimodality type to the required one; too many design variables to consider simultaneously without isolating them: each data modality requires specific assumptions and bespoke development.
    - **Low-quality generated samples** despite extensive sweeping over solvers (steps, orders, *noise schedules*) and training strategies: **outputs only vaguely resembling brain T1s**, with unstable anatomy and artifacts.
    - **Tabular branch sidelined**: poor image results led to the tabular branch design and development being temporarily put aside.
- **Post-mortem analysis**: UNet proved **ill-suited for the architectural exploration** required and too sensitive to diverse details, from implementation to training. Moreover, the VAE compression favored a **“token-based” model** operating directly on latent *patches*.

> Phase-1 conclusion: merely altering the solver (DPM-Solver variants) or training recipes brought no substantial improvement; the problem was structural for our regime and multimodality.
> 

### Phase 2 — **Multimodal DiT** + EDM kerras + TabSyn VAE

### Architectural evolution

After a UNet-centric phase with DPM-Solver, the project migrated to a **multimodal DiT (ViT-style)**, an **EDM (Elucidated Diffusion Models)** framework with separate *time embeddings* per modality and, on the tabular side, a **TabSyn-style VAE** to compress columns into more regular latent tokens. This radical shift aims to better control: (i) image↔tabular **coupling** via **symmetric cross-attention with gating**, (ii) the **pacing** of the two branches (ramps, GradNorm, soft TTUR), and (iii) training robustness (Min-SNR, self-conditioning, plus **EMA**).

### Image branch

Images are compressed by the **VAE** discussed above into 2D latents (standard practice in these models; it reduces effective resolution and makes diffusion more stable/lightweight). The DiT input uses **ViT-style PatchEmbed**, which patchifies and projects into **tokens** with **positional embeddings** and a **[CLS]** token (patchifying shortens sequences; positional aspects preserve geometry; [CLS] adds global-context information for anatomical coherence).

Blocks act as **patch mixers**: **self-attention** aggregates intra-image context even at long range (distant but correlated structures) and, when enabled, **cross-attention** points to a tabular summary so tabular data can influence the image branch. Everything is **modulated by AdaLN** with EDM time embedding and a **gate** controlling conditioning strength (initially low to prevent *negative transfer* while the other branch’s signals are immature, then opening up later).

The **per-patch head** predicts the **D-parameterization** (scaled denoised target) instead of noise—a choice expected to improve **loss conditioning** and stabilize gradients.

In practice, we first consolidate structure/coherence in latent space, then progressively and controllably admit the tabular signal.

### Tabular branch

Each **column (tabular feature) corresponds to 1 token**: categoricals are projected from **tempered logits** (from one-hot with label smoothing) and numerics pass through small MLPs.

The encoder–decoder is a **token-wise Transformer** with **Gaussian latents per token (μ, logσ²)**. Numerics are reconstructed with an **MDN (mixture of Gaussians)** to capture tails/asymmetries and avoid variance flattening; following TabSyn, **KL with free-bits** and **β-anneal** help prevent collapse.

### Inter-branch interaction (mutual conditioning)

Interaction is **bidirectional and controlled**:

- **Symmetric cross-attention**: image tokens can attend to tabular tokens and vice versa. Every sub-layer (self-attn, cross-attn, MLP) is **AdaLN-modulated** with **shift/scale/gate** derived from *(time embedding + pooled representation of the other modality)*. As noted, the **gate** is initialized closed (zero) and opens progressively: coupling grows as signals become reliable.
- **Cross-attn dropout** (training only) to avoid spurious dependencies and enforce within-modality capacity. Also, the **cross-modal gradient** has a **dynamic scale** (*cross_grad_scale*) adapted online based on EMA MSEs to prevent the “stronger” branch from dragging the other when it’s still immature.

## Diffusion process (EDM) and generation

The diffusion process follows **EDM (Elucidated Diffusion Models)**, adapted to the multimodal nature of the problem. As noted, image and tabular do not share a single noise modulator: each modality has its own **σ schedule** with different hyperparameters. **Images** use a more aggressive **Min-SNR γ** to accelerate the transition toward structured, high-contrast signals; **tabular**, by contrast, adopts a more conservative γ that avoids over-rewarding low-variance solutions in early phases. This separation is crucial for handling such different modalities.

In tabular model space we also apply per-feature σ: a vector 𝛼, initialized from standard-deviation ratios, modulates noise dimension-wise and is frozen after warm-up. In VAE-latent tabular mode this is unnecessary because scale is already regularized. To stabilize denoising we use 50% self-conditioning: a “teacher” pass (no gradient) supplies a coherent residual for the training pass. At sampling, we use per-modality CFG (typically higher for images) and **EMA**: after warm-up, EMA weights yield more stable samples.

# Training, conditioning, and multimodal balancing

Training addresses observed asymmetry: empirically we noticed that images start becoming realistic around 30–35k steps, while tabulars already tend toward the mean by ~5k.

Hence a series of tactics:

- a **soft TTUR** regulates the tabular update probability, starting low and rising until ~35k, synchronizing with the image branch.
- in parallel, **GradNorm** with learnable weights. The image loss is annealed (≈15k→45k) to focus on details when the signal matures, while for tabular we delay the full effect of EDM/Min-SNR with a targeted warm-up to avoid variance collapse.

Moreover, coupling across branches uses cross-attention with adaptive gating: branch stability is estimated via EMA MSEs, and more gradient is allowed to flow only when the image is sufficiently defined.

Label conditioning is integral: label-dropout ~30% to train both conditioned and unconditioned modes.

To counteract skewness, under-dispersion, and loss of correlations in generated tabular features, we added anti-collapse regularizations: correlation alignment (CORAL/CoVAR-style) focused on off-diagonals, **image-tab InfoNCE** for semantic coherence, and variance matching on numerics. These terms are applied in series with TTUR and GradNorm.

# Metrics

## Evaluation metrics

### Tabular

- **Statistical Similarity (SURE):** comparison between real and synthetic at the **univariate** level (distributions, tails, skewness) and **multivariate** level (dependencies/correlations), to detect variance compression and structural distortions.
- **Mutual Information (SURE):** estimation and comparison of **pairwise MI** to verify whether informative relationships between features are preserved in generated data.
- **TSTR – Train on Synthetic, Test on Real (SURE):** train a predictive model on **synthetics** and evaluate on **reals**: measures the **downstream utility** of generated tabulars and any loss of discriminative signals.

> Tabular metrics are implemented with Clearbox AI’s SURE library (https://github.com/Clearbox-AI/SURE).
> 

### Images

- **SSIM / MS-SSIM:** structural similarity (single-scale and **multi-scale**), sensitive to **luminance, contrast, and structure**; useful for artifacts and “local morphology.”
- **FID:** istance between **Inception** feature distributions for real vs synthetic; assesses **global quality** and support coverage.

> For image metrics we prepared an operational setup of the Synthetic_Images_Metrics_Toolkit (https://github.com/aiformedresearch/Synthetic_Images_Metrics_Toolkit), used to support SSIM/MS-SSIM/FID and to facilitate batching and reports.
> 

### Multimodal coherence

- **Discriminator Score:** a binary discriminator assesses whether an **(image, table)** **pair** is **consistent** or a mismatch; measures **joint plausibility** beyond marginal qualities.
- **Semantic Similarity Score:** similarity between **image-tab embeddings** (or image-text/tab-description): checks that **tabular phenotype** and **anatomy** convey aligned information.

### Privacy

- **Distance to Closest Record (DCR):** minimum distance between a synthetic and its nearest real record; flags **suspicious proximity** (re-identification risk).
- **Membership Inference Attack (MIA):** controlled statistical attack to probe whether one can infer if a real individual was **in the training set** (membership leakage).
- **Image Clone Detection:** identifies **clones or near-clones** of real images among synthetics (copying/overfitting), even under slight transformations.

# Swarm Learning

The goal of **Swarm Learning** was to train the model **without centralizing** clinical and MRI data, reducing **re-identification** risk and moving toward compliance for sensitive categories like medical data. With the swarm, **data remain on site**: only **model weights/updates** are exchanged, not images or tables. This limits exposure of sensitive information and preserves **local governance** (access policies, audit, logging).

We set up **two GPU nodes** and containerized the stack. The **HPE Swarm Learning** library manages **cluster join**, **synchronization rounds**, and **quorum**; between synchronizations, training proceeds as in a centralized scenario.

Architecturally, we made **DiT+EDM** **swarm-compatible** under the free license **without changing the generation logic**.

# Empirical Results

## Image branch trajectory

As noted, the image branch needs **longer training horizons**, but grows fairly steadily: once past the threshold where samples start to “take shape” (~10–20k steps), the model consolidates **gross geometries and intracranial contrasts** and then refines **ventricular contours, gray/white matter**, and hemispheric symmetries. Some artifacts may still appear, in particular regarding gray-white contrast.

Nevertheless, the model yields **empirically plausible and coherent reconstructions** consistent with expected morphology.

The **diversity** question remains open. The available dataset is **small for the multimodal problem** (≈1000 subjects) and imposes a delicate trade-off: pushing **diffusion parameters** to broaden coverage (e.g., widening the σ distribution, raising σ_max, or tightening Min-SNR) **complicates** EDM training dynamics and can introduce **spurious textures** or early-phase instability; keeping them **conservative** protects local quality but risks **mode contraction**. In practice, the sweet spot balances **stability and anatomical fidelity** against maximal image-space exploration: **inter-subject variance** is visible, but **could be wider**, especially in fine cortical detail and ventricular contours. In other words, **intra-mode quality** is satisfactory, **support coverage** could improve with more data or stronger tuning, **non-trivial** given the context and many moving parts.

The **coupling with the tabular branch** is the other axis of uncertainty. In theory, symmetric cross-attention and shared conditioning **should** provide useful signals (alignment between tabular features and anatomy), but **only if the tab branch is stable** and not under-dispersed. With branches out of sync, coupling may **yield no clear benefit**, be intermittent, or even be detrimental because the model must resolve **more unstable variables** in parallel (differentiated noise schedules, adaptive gates, gradient balancing, per-modality CFG). It is thus plausible that the true advantage of coupling emerges **late** in training, when images are already robust and tabular exits its most fragile phase and thus with perfect finetuning. For this reason, the net effect can be extremely **hard to isolate**.

This picture aligns with the literature: in a **unimodal** regime (images only) the problem is **simpler**, with fewer degrees of freedom, fewer cross-constraints, and often allows pushing **noise parameters** and **sampling schedules** more aggressively without having to protect a second branch.

## Limits of the tabular branch

Despite the training apparatus described (soft TTUR, GradNorm with learnable weights, adaptive-gate cross-attention, EDM/Min-SNR warm-up) and targeted regularizations (CORAL/CoVAR, InfoNCE, variance-matching), the tabular branch keeps showing three recurring symptoms: **skewed distributions with compressed variance**, **attenuation of useful predictive signals**, and **insufficiently preserved inter-feature correlations**. This emerges both in model space and when tabulars pass through the latent VAE.

Hypothesized causes:

1. **Dynamics**: the tabular branch “learns” very quickly (≈5k steps) due to lower complexity and tends to stabilize in **low-variance** regions that easily minimize MSE. TTUR and GradNorm reduce asymmetry with images but don’t eliminate it; the **cross-modal gate** helps protect tabulars from the image branch’s turbulence early on, but it also **postpones** learning of multimodal dependencies that could be beneficial. When coupling really opens (~30–35k), tabular has often already “frozen” margins that are too tight.
2. **Representation/decoder**: in the TabSyn-VAE path, numeric reconstruction via **MDN** may favor **conservative** solutions (near mixture means/modes), especially when control metrics or inspection paths pass through expected statistics of the mixture. The chain **latent VAE → diffusion → decoding** may amplify this tendency toward **under-dispersion**. Weighting choices (e.g., 1/k on categorical logits) and normalizing transforms help balance columns, but may not affect higher-order dependencies.
3. **Hyperparameters**: details of **loss and scheduling** can push, if imperfectly tuned, toward variance collapse. Some studies indicate that in EDM, the combination of **Min-SNR** and **small σ** may over-reward “flat” solutions if **γ** or the tabular **warm-up window** is not perfectly aligned with the dataset; **per-feature MSE** optimizes marginals well but **doesn’t “see”** enough dependency structure, and the regularizers (CORAL/CoVAR, variance-matching, InfoNCE), while helpful, may be **insufficient** to simultaneously maintain realistic tails, discriminative signals, and global correlations—especially if the base setup isn’t well tuned.

In short, while the training design has **stabilized the image branch** and made **coupling more controlled**, the tabular branch remains the bottleneck: **contracted marginals**, **weak signal**, and **non-constructive coupling**.

This picture matches the asymmetry of learning times and the difficulty of balancing the two branches.

### Note on results and code

The empirical results presented so far are **preliminary** and intended to describe the architecture’s behavior and limits. The **code supporting the results** **will be uploaded later**, together with guidance for experiment reproducibility.

Meanwhile, we will continue targeted tests on the **same architecture** with **different hyperparameters** to find a more robust balance in multimodal training. In particular, we will vary (in a controlled way) noise/σ scheduling and **Min-SNR γ**, **CFG** amplitudes per modality, **gating** and cross-attention frequency, **TTUR/GradNorm**, **EMA cadence** and, on the tabular side, VAE posterior temperature (τ) and regularizer weights. This phase is **intrinsically time-consuming**: many choices must be evaluated **at different training stages** and, in several cases, the real effect emerges **only near convergence**.

Consequently, publication of the more **organized and definitive metrics** described earlier depends on these experiments and on consolidating **branch balancing**.

Moreover, this documentation may vary slightly.

### Note on results with another, larger dataset

Given the number of variables to assess, we also experimented with **another dataset much larger in images** but with only **4 tabular features/metadata**; in addition, a label was obtained as a **linear combination** of two other features to simulate a classification problem.

Repeating training mainly to evaluate tabular learning, we observed that the samples obtained, while not faithfully reconstructing the distribution of individual features, maintain **high predictive power**, at most **~10% below** classification on real data, while also **preserving high inter-feature correlation**.

This does not directly answer the question “is the initial dataset too small,” since results with the **same hyperparameters** are not definitive (feature distributions not faithfully reconstructed), and the “artificial” dataset also has far fewer tabular features than the original. Nonetheless, it shows that the architecture, while needing more inquiry, can be considered a **solid starting point** for tackling multimodal generation.