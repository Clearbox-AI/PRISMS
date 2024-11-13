import enum
import math
import numpy as np
import torch as th
import torch.distributed as dist
from einops import rearrange, repeat
from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
from . import dist_util


# TODO: to utility
def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices and
    broadcast them to a desired shape.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: the desired shape to broadcast the extracted values to.
    :return: a tensor of shape `broadcast_shape` with the extracted values broadcasted.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)











class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # Log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.maximum(self.posterior_variance, 1e-20)
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
                        For images: [N x C x H x W]
                        For tabular data: [N x C x F]
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(
            1.0 - self.alphas_cumprod, t, x_start.shape
        )
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the normal noise to add.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(
                self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            )
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        This function calculates the posterior mean and variance given the
        original data `x_start`, the noisy data at timestep `t` (`x_t`), and
        the timestep `t`.

        :param x_start: The original data at timestep 0.
        :param x_t: The noisy data at timestep `t`.
        :param t: The current timestep.
        :return: A tuple containing:
                 - `posterior_mean`: The mean of the posterior distribution.
                 - `posterior_variance`: The variance of the posterior distribution.
                 - `posterior_log_variance_clipped`: The log variance, clipped for numerical stability.
                 All outputs have the same shape as `x_start`.
        """
        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        ), "Shapes of outputs do not match input shapes"
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(
            self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(image_{t-1}, tabular_{t-1} | image_t, tabular_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes the data and a batch of timesteps as input.
        :param x: A dictionary containing:
                  - "image": [N x C x H x W] tensor at time t.
                  - "tabular": [N x C x F] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the x_start prediction before it is used to sample.
                            Applies before clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to pass to the model. This can be used for conditioning.
        :return: A dictionary with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
                 - 'model_predict': the outputs of the model.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B = x["image"].shape[0]
        assert t.shape == (B,), "Timesteps t should have shape (batch_size,)"

        # Get model outputs for image and tabular data
        image_output, tabular_output = model(
            x["image"], x["tabular"], self._scale_timesteps(t), **model_kwargs
        )

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        def get_variance(model_output, x):
            """
            Compute the model mean, variance, and predicted x_start for a given modality.

            :param model_output: The output from the model for a modality.
            :param x: The input data x_t for the modality at time t.
            :return: Tuple of (model_mean, model_variance, model_log_variance, pred_xstart)
            """
            dim = 1  # Channel dimension for both image and tabular data

            if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
                assert model_output.shape[dim] == x.shape[dim] * 2, (
                    f"Expected model output channel dimension to be twice the input, "
                    f"but got {model_output.shape[dim]} and {x.shape[dim]}"
                )
                model_output, model_var_values = th.split(
                    model_output, x.shape[dim], dim=dim
                )
                if self.model_var_type == ModelVarType.LEARNED:
                    model_log_variance = model_var_values
                    model_variance = th.exp(model_log_variance)
                else:
                    min_log = _extract_into_tensor(
                        self.posterior_log_variance_clipped, t, x.shape
                    )
                    max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                    # The model_var_values are scaled to be in [-1, 1].
                    frac = (model_var_values + 1) / 2
                    model_log_variance = frac * max_log + (1 - frac) * min_log
                    model_variance = th.exp(model_log_variance)
            else:
                model_variance, model_log_variance = {
                    ModelVarType.FIXED_LARGE: (
                        np.append(self.posterior_variance[1], self.betas[1:]),
                        np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                    ),
                    ModelVarType.FIXED_SMALL: (
                        self.posterior_variance,
                        self.posterior_log_variance_clipped,
                    ),
                }[self.model_var_type]
                model_variance = _extract_into_tensor(model_variance, t, x.shape)
                model_log_variance = _extract_into_tensor(
                    model_log_variance, t, x.shape
                )

            if self.model_mean_type == ModelMeanType.PREVIOUS_X:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
                )
                model_mean = model_output
            elif self.model_mean_type in [
                ModelMeanType.START_X,
                ModelMeanType.EPSILON,
            ]:
                if self.model_mean_type == ModelMeanType.START_X:
                    pred_xstart = process_xstart(model_output)
                else:
                    # If the model predicts epsilon, compute x_start from epsilon
                    pred_xstart = process_xstart(
                        self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                    )
                model_mean, _, _ = self.q_posterior_mean_variance(
                    x_start=pred_xstart, x_t=x, t=t
                )
            else:
                raise NotImplementedError(self.model_mean_type)

            assert (
                    model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
            ), "Mismatch in shapes of model outputs"

            return model_mean, model_variance, model_log_variance, pred_xstart

        # Compute mean, variance, and predicted x_start for image and tabular data
        image_mean, image_variance, image_log_variance, pred_image_xstart = get_variance(
            image_output, x["image"]
        )
        tabular_mean, tabular_variance, tabular_log_variance, pred_tabular_xstart = get_variance(
            tabular_output, x["tabular"]
        )

        return {
            "mean": {"image": image_mean, "tabular": tabular_mean},
            "variance": {"image": image_variance, "tabular": tabular_variance},
            "log_variance": {"image": image_log_variance, "tabular": tabular_log_variance},
            "pred_xstart": {"image": pred_image_xstart, "tabular": pred_tabular_xstart},
            "model_predict": {"image": image_output, "tabular": tabular_output},
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        """
        Reconstruct x_0 (the original input) from the predicted noise (epsilon) and the noisy input x_t.

        :param x_t: The noisy input at timestep t.
        :param t: The current timestep.
        :param eps: The predicted noise by the model.
        :return: The reconstructed x_0.
        """
        assert x_t.shape == eps.shape, "Shapes of x_t and eps must match"
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        """
        Reconstruct x_0 from x_{t-1} and x_t.

        :param x_t: The noisy input at timestep t.
        :param t: The current timestep.
        :param xprev: The predicted x_{t-1}.
        :return: The reconstructed x_0.
        """
        assert x_t.shape == xprev.shape, "Shapes of x_t and xprev must match"
        return (
                _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
                - _extract_into_tensor(
            self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
        )
                * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        """
        Predict the noise (epsilon) given x_t and the predicted x_0.

        :param x_t: The noisy input at timestep t.
        :param t: The current timestep.
        :param pred_xstart: The predicted x_0.
        :return: The predicted noise (epsilon).
        """
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        """
        Scale the timesteps if rescaling is enabled.

        :param t: The original timesteps.
        :return: The scaled timesteps.
        """
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t





