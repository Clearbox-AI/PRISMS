from multi_modal_diffusion import dist_util, logger
from multi_modal_diffusion.resample import create_named_schedule_sampler
from diffusion_process.dataloaders import load_training_data
from multi_modal_diffusion.model.model_setup import create_model_and_diffusion
from diffusion_process.multimodal_train_util import TrainLoop
from multi_modal_diffusion.configs.defaults import get_default_config
from multi_modal_diffusion.training.pipeline import TrainingStep
from multi_modal_diffusion.common import set_seed_logger_random


class BaseTrainingStep(TrainingStep):
    def __init__(self, num_epochs=20):
        """
        Initialize the base training step with any necessary parameters.

        Args:
            num_epochs (int): Number of epochs to train in this base step.
        """
        self.num_epochs = num_epochs
        self.model = None
        self.diffusion = None
        self.data_loader = None
        self.schedule_sampler = None

    def setup(self, args, model=None, diffusion=None):
        # Set seed and logger
        args = set_seed_logger_random(args)
        logger.configure(args.output_dir)

        # Distributed setup
        dist_util.setup_dist(args.devices if args.devices else "cpu")

        # Load data
        self.data_loader = load_training_data(args)
        sample_batch = next(iter(self.data_loader))
        image_shape = sample_batch['image'].shape
        tabular_shape = sample_batch['tabular'].shape
        args.image_size = ','.join(map(str, image_shape[1:]))
        args.tabular_size = str(tabular_shape[1])

        # Create model and diffusion
        self.model, self.diffusion = create_model_and_diffusion(
            **{k: getattr(args, k) for k in get_default_config().keys()}
        )

        self.model.to(dist_util.dev())
        self.schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, self.diffusion)

    def run(self, args, model=None, diffusion=None):
        # Train from scratch
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
