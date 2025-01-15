import copy
import functools
import os
import glob
import random
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from einops import rearrange, repeat
from PIL import Image
from multi_modal_diffusion import dist_util, logger
from multi_modal_diffusion.fp16_util import MixedPrecisionTrainer
from multi_modal_diffusion.nn import update_ema
from multi_modal_diffusion.resample import LossAwareSampler, UniformSampler
from diffusion_process.multimodal_dpm_solver_plus import DPM_Solver
from diffusion_process.evaluation_metrics import (compute_mmd_tabular, compute_fid, compute_mmd)
from diffusion_process.metrics_utilities import plot_metrics
import time
from diffusers.models import AutoencoderKL

INITIAL_LOG_LOSS_SCALE = 20.0

def debug_memory(prefix=""):
    if th.cuda.is_available():
        allocated = th.cuda.memory_allocated() / (1024**2)
        reserved = th.cuda.memory_reserved() / (1024**2)
        print(f"{prefix} GPU Memory - Allocated: {allocated:.2f}MB, Reserved: {reserved:.2f}MB")
    else:
        print(f"{prefix} GPU Memory Debug: CPU mode, no CUDA available.")

class TrainLoop:
    def __init__(
            self,
            *,
            model,
            diffusion,
            data,
            batch_size,
            microbatch,
            ema_rate,
            log_interval,
            save_interval,
            resume_checkpoint,
            num_epochs,
            lr=0,
            t_lr=1e-4,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=None,
            weight_decay=0.0,
            lr_anneal_steps=0,
            class_cond=False,
            sample_fn='dpm_solver',
            num_classes=0,
            save_row=2,
            eval_interval=1,  # Evaluate every epoch by default
            num_eval_samples=20
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.t_lr = t_lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.num_epochs = num_epochs
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.class_cond = class_cond
        self.num_classes = num_classes
        self.save_row = save_row
        self.step = 1
        self.resume_step = 0
        self.global_batch = self.batch_size * dist_util.get_world_size()
        self.eval_interval = eval_interval
        self.num_eval_samples = num_eval_samples

        self.sync_cuda = th.cuda.is_available()
        self.sample_fn = sample_fn

        print("DEBUG: Initializing TrainLoop")
        # debug_memory("Init start:")

        self._load_and_sync_parameters()

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]
        self.output_model_stastics()

        # Properly enable DDP if in a distributed environment
        if dist.is_initialized() and dist_util.get_world_size() > 1:
            print("DEBUG: Initializing DDP")
            # debug_memory("Before DDP:")
            # Ensure device_ids is an integer index if needed
            local_rank = dist_util.dev().index if dist_util.dev().type == 'cuda' else None
            if local_rank is None:
                local_rank = 0

            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=True,
            )
            dist_util.sync_params(self.ddp_model.parameters())
            # debug_memory("After DDP:")

        else:
            if dist_util.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

        # debug_memory("Init end:")

    def output_model_stastics(self):
        num_params_total = sum(p.numel() for p in self.model.parameters())
        num_params_train = 0
        num_params_pre_load = 0

        for param_group in self.opt.param_groups:
            if param_group['lr'] > 0:
                num_params_train += sum(p.numel() for p in param_group['params'] if p.requires_grad)

        if hasattr(self, 'pre_load_params'):
            num_params_pre_load = sum(
                p.numel() for name, p in self.model.named_parameters() if name in self.pre_load_params)
        if num_params_total > 1e6:
            num_params_total /= 1e6
            num_params_train /= 1e6
            num_params_pre_load /= 1e6
            params_total_label = 'M'
        elif num_params_total > 1e3:
            num_params_total /= 1e3
            num_params_train /= 1e3
            num_params_pre_load /= 1e3
            params_total_label = 'k'
        else:
            params_total_label = ''

        logger.log("Total Parameters: {:.2f}{}".format(num_params_total, params_total_label))
        logger.log("Total Training Parameters: {:.2f}{}".format(num_params_train, params_total_label))
        logger.log("Total Loaded Parameters: {:.2f}{}".format(num_params_pre_load, params_total_label))

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            # Log if in single GPU setup or if it's rank 0 in a distributed setup
            if self.resume_step > 0 and (not dist.is_initialized() or dist.get_rank() == 0):
                logger.log(f"continue training from step {self.resume_step}")
            state_dict = dist_util.load_state_dict(resume_checkpoint, map_location=dist_util.dev())
            self.pre_load_params = state_dict.keys()
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(state_dict)

        # Synchronize parameters only if distributed is initialized
        if dist.is_initialized():
            dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)

        if ema_checkpoint:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                ema_checkpoint, map_location=dist_util.dev()
            )
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = os.path.join(
            os.path.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if os.path.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)


    def run_loop(self):
        for epoch in range(self.num_epochs):
            logger.log(f"Starting epoch {epoch + 1}/{self.num_epochs}")
            # If using a DistributedSampler, set the epoch for proper shuffling.
            if isinstance(self.data.sampler, th.utils.data.DistributedSampler):
                self.data.sampler.set_epoch(epoch)

            for batch in self.data:
                # debug_memory(f"Before run_step (epoch {epoch + 1}, step {self.step}):")
                loss = self.run_step(batch)
                # debug_memory(f"After run_step (epoch {epoch + 1}, step {self.step}):")

                if not dist.is_initialized():
                    print(f"Epoch {epoch + 1}, Step {self.step}, Loss: {loss}")
                else:
                    if dist.get_rank() == 0:
                        print(f"Epoch {epoch + 1}, Step {self.step}, Loss: {loss}")
                # print(f"Epoch {epoch + 1}, Step {self.step}, Loss: {loss}")

                if self.step % self.log_interval == 0:
                    logger.dumpkvs()

                if self.step % self.save_interval == 0:
                    # TODO REMOVED SAVING
                    # self.save()
                    ...
                    # Run for a finite amount of time in integration tests.

                    # TODO: fix save sample
                    self.save_samples()
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return

                self.step += 1

                if self.lr_anneal_steps and self.step + self.resume_step >= self.lr_anneal_steps:
                    logger.log("Reached learning rate annealing steps.")
                    return  # Exit the training loop

            # evaluation step
            if (epoch + 1) % self.eval_interval == 0:
                # debug_memory("Before evaluate_model:")
                # TODO: to fix with 4 channels
                # self.evaluate_model(epoch)
                # debug_memory("After evaluate_model:")
                logger.dumpkvs()

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            # debug_memory("Before final save:")
            self.save()
            # debug_memory("After final save:")

        plot_metrics()

    def run_step(self, batch, cond={}):
        self.mp_trainer.zero_grad()
        loss = self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        return loss



    def evaluate_model(self, epoch):
        # Create a copy of the model for evaluation
        eval_model = copy.deepcopy(self.model)
        eval_model.to(dist_util.dev())  # Move the model copy to the appropriate device

        # Load EMA parameters if needed
        if len(self.ema_params) > 0:
            state_dict = self.mp_trainer.master_params_to_state_dict(self.ema_params[0])
            eval_model.load_state_dict(state_dict)

        eval_model.eval()  # Set model to evaluation mode

        with th.no_grad():
            # Generate samples using eval_model
            generated_images, generated_tabular = self.generate_samples(num_samples=self.num_eval_samples,
                                                                        model=eval_model)
            # Get real samples
            real_images, real_tabular = self.get_real_samples(num_samples=self.num_eval_samples)

            # Postprocess images
            processed_generated_images = self.postprocess_images(generated_images)
            processed_real_images = self.postprocess_images(real_images)

            # Compute metrics
            fid_score = compute_fid(processed_generated_images, processed_real_images)
            mmd_score = compute_mmd(generated_images, real_images)
            mmd_tabular = compute_mmd_tabular(generated_tabular, real_tabular)

            logger.logkv_mean("FID Score", fid_score)
            logger.logkv_mean("MMD Score", mmd_score)
            logger.logkv_mean("Tabular MMD Score", mmd_tabular)

        # No need to reset the training model

        # Dump logs if needed
        logger.dumpkvs()




    def postprocess_images(self, images):
        # Detach from computation graph
        images = images.detach()

        # Flatten the images to compute global min and max
        min_val = images.min()
        max_val = images.max()

        # Handle different possible ranges
        if min_val >= -1.0 and max_val <= 1.0:
            # Images are in [-1, 1], scale to [0, 1]
            images = (images + 1.0) / 2.0
        elif min_val >= 0.0 and max_val <= 1.0:
            # Images are already in [0, 1], no scaling needed
            pass
        else:
            # Images are in an arbitrary range, scale to [0, 1]
            images = (images - min_val) / (max_val - min_val)

        # Ensure images are in [0, 1]
        images = images.clamp(0, 1)

        # Scale images to [0, 255]
        images = (images * 255.0).clamp(0, 255)

        # Convert to uint8
        images = images.type(th.uint8)

        # Ensure images are on the same device as the metric (GPU)
        images = images.to(dist_util.dev())

        return images



    def generate_samples(self, num_samples=20, model=None):
        if model is None:
            model = self.model  # Default to the training model if none is provided

        model.eval()
        with th.no_grad():
            sample_fn = (
                self.diffusion.p_sample_loop if self.sample_fn != 'ddim' else self.diffusion.ddim_sample_loop
            )
            image_size, tabular_size = get_model_sizes(self.model)
            sample = sample_fn(
                model=model,
                shape={
                    # "image": [num_samples, *self.model.modules.image_size],
                    # "tabular": [num_samples, self.model.modules.tabular_size]
                    "image": [num_samples, *image_size],
                    "tabular": [num_samples, tabular_size]
                },
                clip_denoised=True,
            )
            generated_images = sample['image']
            generated_tabular = sample['tabular']
        model.train()
        return generated_images, generated_tabular

    def get_real_samples(self, num_samples=20):
        real_images = []
        real_tabular = []
        num_collected = 0
        for batch in self.data:
            real_images.append(batch['image'])
            real_tabular.append(batch['tabular'])
            num_collected += batch['image'].size(0)
            if num_collected >= num_samples:
                break
        real_images = th.cat(real_images, dim=0)[:num_samples]
        real_tabular = th.cat(real_tabular, dim=0)[:num_samples]
        return real_images.to(dist_util.dev()), real_tabular.to(dist_util.dev())




    def forward_backward(self, batch, cond):
        batch = {k: v.to(dist_util.dev()) for k, v in batch.items()}
        cond = {k: v.to(dist_util.dev()) for k, v in cond.items()}
        batch_len = batch['image'].shape[0]

        for i in range(0, batch_len, self.microbatch):
            micro = {
                k: v[i: i + self.microbatch]
                for k, v in batch.items()
            }

            micro_cond = {
                k: v[i: i + self.microbatch]
                for k, v in cond.items()
            }

            last_batch = (i + self.microbatch) >= batch_len
            t, weights = self.schedule_sampler.sample(self.batch_size, dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.multimodal_training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            loss = (losses["loss"] * weights).mean()
            self.mp_trainer.backward(loss)

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )

        log_loss_dict(
            self.diffusion, t, {k: v * weights for k, v in losses.items()}
        )

        return losses

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)


    def save_samples(self):

        def get_rank_safe():
            return dist.get_rank() if dist.is_initialized() else 0

        def get_world_size_safe():
            return dist.get_world_size() if dist.is_initialized() else 1

        rank = get_rank_safe()
        world_size = get_world_size_safe()

        all_images = []
        all_tabular = []
        logger.log("create samples...")

        # Create a copy of the model for sampling
        sample_model = copy.deepcopy(self.model)
        sample_model.to(dist_util.dev())  # Move the model copy to the appropriate device

        # Use EMA parameters for sampling if available
        if len(self.ema_params) > 0:
            state_dict = self.mp_trainer.master_params_to_state_dict(self.ema_params[0])
            sample_model.load_state_dict(state_dict)

        # Set the sample model to evaluation mode
        sample_model.eval()

        # Safely get image and tabular sizes, supporting both DDP and single-GPU
        image_size, tabular_size = get_model_sizes(self.model)

        total_samples = 0
        while total_samples < self.save_row ** 2:
            model_kwargs = {}

            if self.class_cond:
                classes = th.randint(
                    low=0, high=self.num_classes, size=(self.batch_size,), device=dist_util.dev()
                )
                model_kwargs["y"] = classes

            if self.sample_fn == 'dpm_solver': # the dpm++ one with predict_x0=True, thresholding=False
                dpm_solver = DPM_Solver(
                    model=self.model,
                    alphas_cumprod=th.tensor(self.diffusion.alphas_cumprod, device=dist_util.dev())
                )
                x_T = {
                    "image": th.randn([self.batch_size, *image_size], device=dist_util.dev()),
                    "tabular": th.randn([self.batch_size, tabular_size], device=dist_util.dev())
                }
                sample = dpm_solver.sample(
                    x_T,
                    steps=20,
                    order=2,
                    skip_type="logSNR",
                    method="adaptive",
                )
            else:
                sample_fn = (
                    self.diffusion.p_sample_loop if self.sample_fn != 'ddim' else self.diffusion.ddim_sample_loop
                )
                sample = sample_fn(
                    model=self.model,  # Using the original model as in the original code
                    shape={
                        "image": [self.batch_size, *image_size],
                        "tabular": [self.batch_size, tabular_size]
                    },
                    clip_denoised=True,
                    model_kwargs=model_kwargs,
                )

            sample_image = sample['image']
            sample_tabular = sample['tabular']

            device = th.device("cuda" if th.cuda.is_available() else "cpu")
            autoencoder = AutoencoderKL.from_pretrained("/home/PRISMS/model_garden/microsoft_mri_autoencoder_01/weights").to(device)
            autoencoder.eval()


            with th.no_grad():
                decoded = autoencoder.decode(sample_image.to(device)).sample

            decoded_np = decoded.detach().cpu().numpy()

            import matplotlib.pyplot as plt
            plt.imshow(decoded_np[0,0], cmap="gray")



            # TODO: TRY
            # Convert from [-1, 1] to [0, 255]
            # sample_image = ((sample_image + 1) * 127.5).clamp(0, 255).to(th.uint8)
            # Normalize to [0, 1]
            sample_image = (sample_image - sample_image.min()) / (sample_image.max() - sample_image.min()) * 255
            sample_image = sample_image.to(th.uint8)


            # If in distributed mode, gather samples from all ranks
            if world_size > 1:
                gathered_sample_images = [th.zeros_like(sample_image) for _ in range(world_size)]
                dist.all_gather(gathered_sample_images, sample_image)
            else:
                gathered_sample_images = [sample_image]

            all_images.extend([img_tensor.cpu().numpy() for img_tensor in gathered_sample_images])

            if world_size > 1:
                gathered_sample_tabular = [th.zeros_like(sample_tabular) for _ in range(world_size)]
                dist.all_gather(gathered_sample_tabular, sample_tabular)
            else:
                gathered_sample_tabular = [sample_tabular]

            all_tabular.extend([tab_tensor.cpu().numpy() for tab_tensor in gathered_sample_tabular])

            total_samples += self.batch_size * world_size

            if rank == 0:
                logger.log(f"{total_samples} samples generated.")

        all_images = np.concatenate(all_images, axis=0)
        all_tabular = np.concatenate(all_tabular, axis=0)

        if rank == 0:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            # Create new folder to save samples
            samples_dir = os.path.join(logger.get_dir(), f'samples_{timestamp}')
            os.makedirs(samples_dir, exist_ok=True)
            # Save images
            for idx, img_array in enumerate(all_images):
                img = Image.fromarray(img_array.transpose(1, 2, 0))
                img.save(os.path.join(samples_dir, f"sample_image_{idx}.png"))

            # Save tabular data
            np.save(os.path.join(samples_dir, f"sample_tabular.npy"), all_tabular)

        if world_size > 1:
            dist.barrier()


    # def save_samples(self):
    #
    #     def get_rank_safe():
    #         return dist.get_rank() if dist.is_initialized() else 0
    #
    #     def get_world_size_safe():
    #         return dist.get_world_size() if dist.is_initialized() else 1
    #
    #     rank = get_rank_safe()
    #     world_size = get_world_size_safe()
    #
    #     all_images = []
    #     all_tabular = []
    #     logger.log("create samples...")
    #
    #     # Create a copy of the model for sampling
    #     sample_model = copy.deepcopy(self.model)
    #     sample_model.to(dist_util.dev())  # Move the model copy to the appropriate device
    #
    #     # Use EMA parameters for sampling if available
    #     if len(self.ema_params) > 0:
    #         state_dict = self.mp_trainer.master_params_to_state_dict(self.ema_params[0])
    #         sample_model.load_state_dict(state_dict)
    #
    #     # Set the sample model to evaluation mode
    #     sample_model.eval()
    #
    #     # Safely get image and tabular sizes, supporting both DDP and single-GPU
    #     image_size, tabular_size = get_model_sizes(self.model)
    #
    #     total_samples = 0
    #     while total_samples < self.save_row ** 2:
    #         model_kwargs = {}
    #
    #         if self.class_cond:
    #             classes = th.randint(
    #                 low=0, high=self.num_classes, size=(self.batch_size,), device=dist_util.dev()
    #             )
    #             model_kwargs["y"] = classes
    #
    #         if self.sample_fn == 'dpm_solver':
    #             dpm_solver = DPM_Solver(
    #                 model=self.model,  # Using the original model here as in the original code
    #                 alphas_cumprod=th.tensor(self.diffusion.alphas_cumprod, device=dist_util.dev())
    #             )
    #             x_T = {
    #                 "image": th.randn([self.batch_size, *image_size], device=dist_util.dev()),
    #                 "tabular": th.randn([self.batch_size, tabular_size], device=dist_util.dev())
    #             }
    #             sample = dpm_solver.sample(
    #                 x_T,
    #                 steps=20,
    #                 order=2,
    #                 skip_type="logSNR",
    #                 method="adaptive",
    #             )
    #         else:
    #             sample_fn = (
    #                 self.diffusion.p_sample_loop if self.sample_fn != 'ddim' else self.diffusion.ddim_sample_loop
    #             )
    #             sample = sample_fn(
    #                 model=self.model,  # Using the original model as in the original code
    #                 shape={
    #                     "image": [self.batch_size, *image_size],
    #                     "tabular": [self.batch_size, tabular_size]
    #                 },
    #                 clip_denoised=True,
    #                 model_kwargs=model_kwargs,
    #             )
    #
    #         sample_image = sample['image']
    #         sample_tabular = sample['tabular']
    #
    #         # Convert from [-1, 1] to [0, 255]
    #         sample_image = ((sample_image + 1) * 127.5).clamp(0, 255).to(th.uint8)
    #
    #         # If in distributed mode, gather samples from all ranks
    #         if world_size > 1:
    #             gathered_sample_images = [th.zeros_like(sample_image) for _ in range(world_size)]
    #             dist.all_gather(gathered_sample_images, sample_image)
    #         else:
    #             gathered_sample_images = [sample_image]
    #
    #         all_images.extend([img_tensor.cpu().numpy() for img_tensor in gathered_sample_images])
    #
    #         if world_size > 1:
    #             gathered_sample_tabular = [th.zeros_like(sample_tabular) for _ in range(world_size)]
    #             dist.all_gather(gathered_sample_tabular, sample_tabular)
    #         else:
    #             gathered_sample_tabular = [sample_tabular]
    #
    #         all_tabular.extend([tab_tensor.cpu().numpy() for tab_tensor in gathered_sample_tabular])
    #
    #         total_samples += self.batch_size * world_size
    #
    #         if rank == 0:
    #             logger.log(f"{total_samples} samples generated.")
    #
    #     all_images = np.concatenate(all_images, axis=0)
    #     all_tabular = np.concatenate(all_tabular, axis=0)
    #
    #     if rank == 0:
    #         timestamp = time.strftime('%Y%m%d_%H%M%S')
    #         # Create new folder to save samples
    #         samples_dir = os.path.join(logger.get_dir(), f'samples_{timestamp}')
    #         os.makedirs(samples_dir, exist_ok=True)
    #         # Save images
    #         for idx, img_array in enumerate(all_images):
    #             img = Image.fromarray(img_array.transpose(1, 2, 0))
    #             img.save(os.path.join(samples_dir, f"sample_image_{idx}.png"))
    #
    #         # Save tabular data
    #         np.save(os.path.join(samples_dir, f"sample_tabular.npy"), all_tabular)
    #
    #     if world_size > 1:
    #         dist.barrier()



    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if not dist.is_initialized() or dist.get_rank() == 0:
                # Only the primary process saves checkpoints
                logger.log(f"saving model {rate}...")
                filename = f"model{(self.step + self.resume_step):06d}.pt" if rate == 0 else f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"
                with open(os.path.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if not dist.is_initialized() or dist.get_rank() == 0:
            with open(os.path.join(get_blob_logdir(), f"opt{(self.step + self.resume_step):06d}.pt"), "wb") as f:
                th.save(self.opt.state_dict(), f)

        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    filename = "model*.pt"
    max_step = 0
    for name in glob.glob(os.path.join(get_blob_logdir(), filename)):
        step = int(name[-9:-3])
        max_step = max(max_step, step)
    if max_step:
        path = os.path.join(get_blob_logdir(), f"model{(max_step):06d}.pt")
        if os.path.exists(path):
            return path
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = os.path.join(os.path.dirname(main_checkpoint), filename)
    if os.path.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def get_model_sizes(model):
    # If using DDP
    if hasattr(model, 'module'):
        return model.module.image_size, model.module.tabular_size
    # If running on a single GPU or CPU without DDP
    return model.image_size, model.tabular_size