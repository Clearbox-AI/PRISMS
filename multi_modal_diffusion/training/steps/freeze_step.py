import os
import torch as th
from multi_modal_diffusion import dist_util, logger
from multi_modal_diffusion.resample import create_named_schedule_sampler
from diffusion_process.dataloaders import load_training_data
from diffusion_process.multimodal_train_util import TrainLoop
from multi_modal_diffusion.training.pipeline import TrainingStep
from multi_modal_diffusion.architecture_utils.layers_classification import classify_parameters
from multi_modal_diffusion.architecture_utils.layers_initialization import reinitialize_specific_layers

class FreezeStep(TrainingStep):
    def __init__(self, restore_model_path, freeze_layers='common', num_epochs=20):
        self.restore_model_path = restore_model_path
        self.freeze_layers = freeze_layers
        self.num_epochs = num_epochs

    def get_checkpoints(self, restore_path):
        subnames = ["ema", "model", "opt"]
        latest_files = {sn: None for sn in subnames}
        for sn in subnames:
            matching_files = [f for f in os.listdir(restore_path) if sn in f]
            matching_files.sort()
            if matching_files:
                latest_files[sn] = os.path.join(restore_path, matching_files[-1])
        return latest_files["ema"], latest_files["model"], latest_files["opt"]

    def setup(self, args, model, diffusion):
        # We assume model and diffusion come from previous step
        logger.configure(args.output_dir)
        dist_util.setup_dist(args.devices if args.devices else "cpu")

        self.data_loader = load_training_data(args)

        EMA_CKPT, MODEL_CKPT, _ = self.get_checkpoints(self.restore_model_path)
        state_dict = th.load(MODEL_CKPT, map_location=dist_util.dev())
        model.load_state_dict(state_dict)

        if EMA_CKPT and os.path.exists(EMA_CKPT):
            ema_state = th.load(EMA_CKPT, map_location=dist_util.dev())
            model.load_state_dict(ema_state)

        layer_classification = classify_parameters(model)
        if self.freeze_layers in layer_classification:
            layers_to_reinit = layer_classification[self.freeze_layers]
            reinitialize_specific_layers(model, layers_to_reinit)

        self.schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
        self.model = model
        self.diffusion = diffusion

    def run(self, args, model=None, diffusion=None):
        TrainLoop(
            model=self.model,
            diffusion=self.diffusion,
            data=self.data_loader,
            batch_size=args.batch_size,
            microbatch=args.microbatch,
            ema_rate=args.ema_rate,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            resume_checkpoint=args.resume_checkpoint,
            num_epochs=self.num_epochs,
            lr=args.lr,
            t_lr=args.t_lr,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
            schedule_sampler=self.schedule_sampler,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.lr_anneal_steps,
            class_cond=args.class_cond,
            sample_fn=args.sample_fn,
            eval_interval=1,
            num_eval_samples=20,
        ).run_loop()

        return self.model, self.diffusion
