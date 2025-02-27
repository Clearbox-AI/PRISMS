class TrainingStep:
    """
    A base interface for a training step. Subclasses should implement:
    - setup(args, model, diffusion)
    - run(args, model, diffusion)
    """
    def setup(self, args, model=None, diffusion=None):
        # Setup data, possibly modify model, diffusion
        raise NotImplementedError

    def run(self, args, model=None, diffusion=None):
        # Perform training or adjustments
        raise NotImplementedError


class Pipeline:
    def __init__(self, steps):
        """
        steps: a list of TrainingStep instances
        """
        self.steps = steps

    def run(self, args):
        model = None
        diffusion = None
        for step in self.steps:
            step.setup(args, model, diffusion)
            model, diffusion = step.run(args, model, diffusion)
        return model, diffusion
