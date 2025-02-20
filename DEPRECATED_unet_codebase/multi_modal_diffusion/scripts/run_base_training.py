from multi_modal_diffusion.configs.defaults import create_argparser
from multi_modal_diffusion.training.pipeline import Pipeline
from multi_modal_diffusion.training.steps.base_step import BaseTrainingStep

if __name__ == "__main__":
    parser = create_argparser()
    args = parser.parse_args()

    pipeline = Pipeline(steps=[BaseTrainingStep(num_epochs=200)])
    pipeline.run(args)
