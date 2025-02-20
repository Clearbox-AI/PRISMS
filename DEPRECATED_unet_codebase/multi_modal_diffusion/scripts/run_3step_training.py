from multi_modal_diffusion.configs.defaults import create_argparser
from multi_modal_diffusion.training.pipeline import Pipeline
from multi_modal_diffusion.training.steps.base_step import BaseTrainingStep
from multi_modal_diffusion.training.steps.freeze_step import FreezeStep

RESTORE_MODEL_PATH = "/mnt/storage/lumir_three_stage_exp/stage_one"

if __name__ == "__main__":
    parser = create_argparser()
    args = parser.parse_args()

    # For demonstration, we do a pipeline: Base -> Freeze -> Base again (just as an example)
    pipeline = Pipeline(steps=[
        BaseTrainingStep(num_epochs=20),
        FreezeStep(restore_model_path=RESTORE_MODEL_PATH, freeze_layers='common', num_epochs=20),
        # Add more steps as needed for a 3 step pipeline
    ])
    pipeline.run(args)
