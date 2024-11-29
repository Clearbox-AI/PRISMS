"""
This code is adapted from guided_diffusion:
https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py
Modified for image and tabular multimodal diffusion.
"""

import enum
import math
import numpy as np
import torch as th
import torch.distributed as dist
from einops import rearrange, repeat
from multi_modal_diffusion.nn import mean_flat
from multi_modal_diffusion.losses import normal_kl, discretized_gaussian_log_likelihood
from multi_modal_diffusion import dist_util


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """

    if schedule_name == "linear":
        # Linear schedule from Ho et al., extended to work for any number of diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"Unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    """
    Types of loss functions.
    """

    # TODO: maybe implement tabsyn
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


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
        # Log variance clipped because the posterior variance is 0 at the beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
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

        :param x_start: the [N x F x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """

        mean = (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
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
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """

        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """

        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
            self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(image_{t-1}, tabular_{t-1} | image_t, tabular_t), as well as a prediction of x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: {"image": [N x C x H x W], "tabular": [N x F]}  at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
                 - 'model_outputs': the outputs of the model.
        """

        if model_kwargs is None:
            model_kwargs = {}

        B = x["image"].shape[0]
        assert t.shape == (B,)

        image_output, tabular_output = model(x["image"], x["tabular"], self._scale_timesteps(t), **model_kwargs)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        def get_variance(model_output, x):
            # TODO: to check
            if x.dim() == 2:  # Tabular data
                dim = 1
            elif x.dim() == 4:  # Image data
                dim = 1
            else:
                raise ValueError("Unsupported data dimension: {}".format(x.dim()))
            if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
                assert model_output.shape[dim] == x.shape[dim] * 2
                model_output, model_var_values = th.split(model_output, x.shape[dim], dim=dim)
                if self.model_var_type == ModelVarType.LEARNED:
                    model_log_variance = model_var_values
                    model_variance = th.exp(model_log_variance)
                else:
                    min_log = _extract_into_tensor(
                        self.posterior_log_variance_clipped, t, x.shape
                    )
                    max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                    # The model_var_values is [-1, 1] for [min_var, max_var].
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
                model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

            if self.model_mean_type == ModelMeanType.PREVIOUS_X:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
                )
                model_mean = model_output
            elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
                if self.model_mean_type == ModelMeanType.START_X:
                    pred_xstart = process_xstart(model_output)
                else:
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
            )
            return model_mean, model_variance, model_log_variance, pred_xstart

        image_mean, image_variance, image_log_variance, pred_image_xstart = get_variance(image_output, x["image"])
        tabular_mean, tabular_variance, tabular_log_variance, pred_tabular_xstart = get_variance(tabular_output,
                                                                                                 x["tabular"])

        return {
            "mean": {"image": image_mean, "tabular": tabular_mean},
            "variance": {"image": image_variance, "tabular": tabular_variance},
            "log_variance": {"image": image_log_variance, "tabular": tabular_log_variance},
            "pred_xstart": {"image": pred_image_xstart, "tabular": pred_tabular_xstart},
            "model_predict": {"image": image_output, "tabular": tabular_output}
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (
                _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
                - _extract_into_tensor(
            self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
        )
                * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        """
        Scale timesteps.
        """
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """

        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
                p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        # TODO: to check
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).

        Adapted for image and tabular data.
        """

        if model_kwargs is None:
            model_kwargs = {}

        out = {
            "mean": {},
            "variance": p_mean_var["variance"],
            "log_variance": p_mean_var["log_variance"],
            "pred_xstart": {},
            "model_predict": p_mean_var["model_predict"]
        }

        for modality in ["image", "tabular"]:
            x_modality = x[modality]
            t_modality = t
            p_mean_var_modality = {
                "mean": p_mean_var["mean"][modality],
                "variance": p_mean_var["variance"][modality],
                "log_variance": p_mean_var["log_variance"][modality],
                "pred_xstart": p_mean_var["pred_xstart"][modality],
                "model_predict": p_mean_var["model_predict"][modality]
            }

            alpha_bar = _extract_into_tensor(self.alphas_cumprod, t_modality, x_modality.shape)

            eps = self._predict_eps_from_xstart(x_modality, t_modality, p_mean_var_modality["pred_xstart"])
            eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
                x_modality, self._scale_timesteps(t_modality), modality=modality, **model_kwargs
            )

            # Update pred_xstart
            out["pred_xstart"][modality] = self._predict_xstart_from_eps(x_modality, t_modality, eps)

            # Recompute mean
            mean_modality, _, _ = self.q_posterior_mean_variance(
                x_start=out["pred_xstart"][modality], x_t=x_modality, t=t_modality
            )
            out["mean"][modality] = mean_modality

        return out

    def p_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            noise=None
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor dict at x_{t-1}: {"image":[N,C,H,W]; "tabular":[N,F]}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_start': a prediction of x_0.
                 - 'pred_noise': a prediction of epsilon.
        """

        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        image_noise = th.randn_like(x["image"])
        tabular_noise = th.randn_like(x["tabular"])

        image_nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x["image"].shape) - 1)))
        )  # no noise when t == 0

        tabular_nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x["tabular"].shape) - 1)))
        )  # no noise when t == 0

        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )

        image_sample = out["mean"]["image"] + image_nonzero_mask * th.exp(
            0.5 * out["log_variance"]["image"]) * image_noise
        tabular_sample = out["mean"]["tabular"] + tabular_nonzero_mask * th.exp(
            0.5 * out["log_variance"]["tabular"]) * tabular_noise

        return {
            "sample": {"image": image_sample, "tabular": tabular_sample},
            "pred_start": {"image": out["pred_xstart"]["image"], "tabular": out["pred_xstart"]["tabular"]},
            "pred_noise": {"image": out["model_predict"]["image"], "tabular": out["model_predict"]["tabular"]}
        }

    def p_sample_loop(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=True
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, {'image':(N, F, H, W), 'tabular':(N, F)}
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """

        final = None
        for sample in self.p_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress
        ):
            final = sample
        return final

    def p_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """

        if device is None:
            device = dist_util.dev()

        image = th.randn(*shape["image"], device=device)
        tabular = th.randn(*shape["tabular"], device=device)
        x = {"image": image, "tabular": tabular}
        indices = list(range(self.num_timesteps))[::-1]  # From 0 to 999

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            cond = None
            t = th.tensor([i] * shape["image"][0], device=device)

            with th.no_grad():
                out = self.p_sample(
                    model,
                    x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond,
                    model_kwargs=model_kwargs,
                    noise=noise,
                )
                yield out["sample"]
                x = out["sample"]

    def conditional_p_sample_loop(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=True,
            class_scale=0.0,
    ):
        """
        Zero-shot conditional generation of samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, {"image": (N, C, H, W), "tabular": (N, F)}.
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param class_scale: if 0, replacement method is used. Otherwise, the gradient-based method is used.
        :return: a non-differentiable batch of samples.
        """

        final = None
        if class_scale == 0:
            conditional_p_sample_loop_progressive_func = self.conditional_p_sample_loop_progressive_unscale
        else:
            conditional_p_sample_loop_progressive_func = self.conditional_p_sample_loop_progressive_scale

        for sample in conditional_p_sample_loop_progressive_func(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                class_scale=class_scale
        ):
            final = sample

        return final

    def conditional_p_sample_loop_progressive_unscale(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            class_scale=0.0,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion using the unscaled method.

        Arguments are the same as conditional_p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """

        if device is None:
            device = next(model.parameters()).device

        if noise is None:
            image = th.randn(*shape["image"], device=device)
            tabular = th.randn(*shape["tabular"], device=device)
            noise = {"image": image, "tabular": tabular}

        x = noise.copy()

        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        image_condition = model_kwargs.pop("image", None)
        tabular_condition = model_kwargs.pop("tabular", None)

        for i in indices:
            cond = cond_fn if cond_fn is not None else None
            t = th.tensor([i] * shape["image"][0], device=device)

            if image_condition is not None:
                x["image"] = self.q_sample(image_condition, t, noise=noise["image"])

            if tabular_condition is not None:
                x["tabular"] = self.q_sample(tabular_condition, t, noise=noise["tabular"])

            with th.no_grad():
                out = self.p_sample(
                    model,
                    x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond,
                    model_kwargs=model_kwargs,
                    noise=noise,
                )
                yield out["sample"]
                x = out["sample"]

    def conditional_p_sample_loop_progressive_scale(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            class_scale=3.0,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion using the scaled method.

        Arguments are the same as conditional_p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """

        if device is None:
            device = next(model.parameters()).device

        if noise is None:
            image = th.randn(*shape["image"], device=device)
            tabular = th.randn(*shape["tabular"], device=device)
            noise = {"image": image, "tabular": tabular}

        x = noise.copy()

        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        image_condition = model_kwargs.pop("image", None)
        tabular_condition = model_kwargs.pop("tabular", None)

        for i in indices:
            cond = cond_fn if cond_fn is not None else None
            t = th.tensor([i] * shape["image"][0], device=device)

            # Identify conditioned and target modalities
            conditioned_modality = None
            target_modality = None

            if image_condition is not None:
                conditioned_modality = "image"
                target_modality = "tabular"
                x["image"] = self.q_sample(image_condition, t, noise=noise["image"])
                previous_step_condition = self.q_sample(image_condition, t - 1, noise=noise["image"])

            if tabular_condition is not None:
                conditioned_modality = "tabular"
                target_modality = "image"
                x["tabular"] = self.q_sample(tabular_condition, t, noise=noise["tabular"])
                previous_step_condition = self.q_sample(tabular_condition, t - 1, noise=noise["tabular"])

            with th.enable_grad():
                none_zero_mask = (t != 0).float().view(-1, *([1] * (len(x[target_modality].shape) - 1)))
                x[target_modality] = x[target_modality].detach().requires_grad_()
                out = self.p_sample(
                    model,
                    x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond,
                    model_kwargs=model_kwargs,
                    noise=noise,
                )

                previous_step_pred = out["sample"]

                # Compute loss between predicted conditioned modality and its previous step
                loss = mean_flat((previous_step_pred[conditioned_modality] - previous_step_condition) ** 2)
                loss_scale = 1.0
                grad = th.autograd.grad(loss.mean() * loss_scale, x[target_modality])[0]

                x[target_modality] = previous_step_pred[target_modality] - none_zero_mask * grad * class_scale * \
                                     self.sqrt_alphas_cumprod[i]

            yield x

    def ddim_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Re-derive epsilon if needed
        eps = {}
        eps["image"] = self._predict_eps_from_xstart(x["image"], t, out["pred_xstart"]["image"])
        eps["tabular"] = self._predict_eps_from_xstart(x["tabular"], t, out["pred_xstart"]["tabular"])

        alpha_bar = {}
        alpha_bar["image"] = _extract_into_tensor(self.alphas_cumprod, t, x["image"].shape)
        alpha_bar["tabular"] = _extract_into_tensor(self.alphas_cumprod, t, x["tabular"].shape)

        alpha_bar_prev = {}
        alpha_bar_prev["image"] = _extract_into_tensor(self.alphas_cumprod_prev, t, x["image"].shape)
        alpha_bar_prev["tabular"] = _extract_into_tensor(self.alphas_cumprod_prev, t, x["tabular"].shape)

        sigma = {}
        sigma["image"] = (
                eta * th.sqrt((1 - alpha_bar_prev["image"]) / (1 - alpha_bar["image"]))
                * th.sqrt(1 - alpha_bar["image"] / alpha_bar_prev["image"])
        )
        sigma["tabular"] = (
                eta * th.sqrt((1 - alpha_bar_prev["tabular"]) / (1 - alpha_bar["tabular"]))
                * th.sqrt(1 - alpha_bar["tabular"] / alpha_bar_prev["tabular"])
        )

        # Equation 12
        noise = {}
        noise["image"] = th.randn_like(x["image"])
        noise["tabular"] = th.randn_like(x["tabular"])

        mean_pred = {}
        mean_pred["image"] = (
                out["pred_xstart"]["image"] * th.sqrt(alpha_bar_prev["image"])
                + th.sqrt(1 - alpha_bar_prev["image"] - sigma["image"] ** 2) * eps["image"]
        )
        mean_pred["tabular"] = (
                out["pred_xstart"]["tabular"] * th.sqrt(alpha_bar_prev["tabular"])
                + th.sqrt(1 - alpha_bar_prev["tabular"] - sigma["tabular"] ** 2) * eps["tabular"]
        )

        nonzero_mask = {}
        nonzero_mask["image"] = (
            (t != 0).float().view(-1, *([1] * (len(x["image"].shape) - 1)))
        )
        nonzero_mask["tabular"] = (
            (t != 0).float().view(-1, *([1] * (len(x["tabular"].shape) - 1)))
        )

        sample = {}
        sample["image"] = mean_pred["image"] + nonzero_mask["image"] * sigma["image"] * noise["image"]
        sample["tabular"] = mean_pred["tabular"] + nonzero_mask["tabular"] * sigma["tabular"] * noise["tabular"]

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """

        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        eps = {}
        eps["image"] = (
                               _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x["image"].shape) * x["image"]
                               - out["pred_xstart"]["image"]
                       ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x["image"].shape)

        eps["tabular"] = (
                                 _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x["tabular"].shape) * x[
                             "tabular"]
                                 - out["pred_xstart"]["tabular"]
                         ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x["tabular"].shape)

        alpha_bar_next = {}
        alpha_bar_next["image"] = _extract_into_tensor(self.alphas_cumprod_next, t, x["image"].shape)
        alpha_bar_next["tabular"] = _extract_into_tensor(self.alphas_cumprod_next, t, x["tabular"].shape)

        mean_pred = {}
        mean_pred["image"] = (
                out["pred_xstart"]["image"] * th.sqrt(alpha_bar_next["image"])
                + th.sqrt(1 - alpha_bar_next["image"]) * eps["image"]
        )
        mean_pred["tabular"] = (
                out["pred_xstart"]["tabular"] * th.sqrt(alpha_bar_next["tabular"])
                + th.sqrt(1 - alpha_bar_next["tabular"]) * eps["tabular"]
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        eps = {}
        eps["image"] = (
                               _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x["image"].shape) * x["image"]
                               - out["pred_xstart"]["image"]
                       ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x["image"].shape)

        eps["tabular"] = (
                                 _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x["tabular"].shape) * x[
                             "tabular"]
                                 - out["pred_xstart"]["tabular"]
                         ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x["tabular"].shape)

        alpha_bar_next = {}
        alpha_bar_next["image"] = _extract_into_tensor(self.alphas_cumprod_next, t, x["image"].shape)
        alpha_bar_next["tabular"] = _extract_into_tensor(self.alphas_cumprod_next, t, x["tabular"].shape)

        mean_pred = {}
        mean_pred["image"] = (
                out["pred_xstart"]["image"] * th.sqrt(alpha_bar_next["image"])
                + th.sqrt(1 - alpha_bar_next["image"]) * eps["image"]
        )
        mean_pred["tabular"] = (
                out["pred_xstart"]["tabular"] * th.sqrt(alpha_bar_next["tabular"])
                + th.sqrt(1 - alpha_bar_next["tabular"]) * eps["tabular"]
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=True,
            eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                eta=eta
        ):
            final = sample
        return final

    def ddim_sample_loop(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=True,
            eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """

        final = None
        for sample in self.ddim_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                eta=eta
        ):
            final = sample
        return final

    def ddim_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """

        if device is None:
            device = next(model.parameters()).device

        if noise is None:
            image = th.randn(*shape["image"], device=device)
            tabular = th.randn(*shape["tabular"], device=device)
            x = {"image": image, "tabular": tabular}
        else:
            x = noise.copy()

        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape["image"][0], device=device)
            cond = cond_fn if cond_fn is not None else None

            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out["sample"]
                x = out["sample"]

    def _vb_terms_bpd(
            self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """

        image_true_mean, _, image_true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start["image"], x_t=x_t["image"], t=t
        )
        tabular_true_mean, _, tabular_true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start["tabular"], x_t=x_t["tabular"], t=t
        )
        true_mean = {"image": image_true_mean, "tabular": tabular_true_mean}
        true_log_variance_clipped = {"image": image_true_log_variance_clipped,
                                     "tabular": tabular_true_log_variance_clipped}

        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )

        kl = {}
        decoder_nll = {}
        output = {}
        for key in ["image", "tabular"]:
            kl[key] = normal_kl(
                true_mean[key], true_log_variance_clipped[key], out["mean"][key], out["log_variance"][key]
            )
            kl[key] = mean_flat(kl[key]) / np.log(2.0)

            decoder_nll[key] = -discretized_gaussian_log_likelihood(
                x_start[key], means=out["mean"][key], log_scales=0.5 * out["log_variance"][key]
            )
            assert decoder_nll[key].shape == x_start[key].shape
            decoder_nll[key] = mean_flat(decoder_nll[key]) / np.log(2.0)

            output[key] = th.where((t == 0), decoder_nll[key], kl[key])
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def multimodal_training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        image_start = x_start['image']
        tabular_start = x_start['tabular']
        if model_kwargs is None:
            model_kwargs = {}

        if noise is None:
            noise = {
                "image": th.randn_like(image_start),
                "tabular": th.randn_like(tabular_start)
            }
        # 0 means t_th step, 1 means the tabular gives groundtruth, 2 means the image gives the groundtruth

        image_t = self.q_sample(image_start, t, noise=noise["image"])
        tabular_t = self.q_sample(tabular_start, t, noise=noise["tabular"])

        image_output, tabular_output = model(image_t, tabular_t, self._scale_timesteps(t), **model_kwargs)

        logger.logkv("step", tabular_output)
        image_loss = {}
        tabular_loss = {}
        if self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
                image_output, image_var_values = th.split(image_output, image_start.shape[1], dim=1)
                tabular_output, tabular_var_values = th.split(tabular_output, tabular_start.shape[1], dim=1)
                # Learn the variance using the variational bound, but don't let it affect our mean prediction.
                image_frozen_out = th.cat([image_output.detach(), image_var_values], dim=1)
                tabular_frozen_out = th.cat([tabular_output.detach(), tabular_var_values], dim=1)
                frozen_out = {"image": image_frozen_out, "tabular": tabular_frozen_out}
                x_t = {"image": image_t, "tabular": tabular_t}
                vb_loss = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: [r["image"], r["tabular"]],
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                image_loss["vb"] = vb_loss["image"]
                tabular_loss["vb"] = vb_loss["tabular"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    image_loss["vb"] *= self.num_timesteps / 1000.0
                    tabular_loss["vb"] *= self.num_timesteps / 1000.0

            image_target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=image_start, x_t=image_t, t=t
                )[0],
                ModelMeanType.START_X: image_start,
                ModelMeanType.EPSILON: noise["image"],  # noise
            }[self.model_mean_type]
            tabular_target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=tabular_start, x_t=tabular_t, t=t
                )[0],
                ModelMeanType.START_X: tabular_start,
                ModelMeanType.EPSILON: noise["tabular"],  # noise
            }[self.model_mean_type]

            image_loss["mse"] = mean_flat((image_target - image_output) ** 2)
            tabular_loss["mse"] = mean_flat((tabular_target - tabular_output) ** 2)

        term = {"loss": 0}

        for key in image_loss.keys():
            term[f"{key}_image"] = image_loss[key]
            term[f"{key}_tabular"] = tabular_loss[key]
            term["loss"] += term[f"{key}_image"] + term[f"{key}_tabular"]

        return term

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """

        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim.
        """

        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """

    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
