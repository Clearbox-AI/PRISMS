# evaluation/metrics/manager.py
import hydra
from omegaconf import DictConfig
from typing import Dict, Any

from evaluation.metrics.FID import FIDMetric


# from evaluation.metrics.IS import ISMetric
# from evaluation.metrics.SSIM import SSIMMetric
# ...

def instantiate_metrics(cfg: DictConfig, shared_state: Dict[str, Any]):
    """
    Given the Hydra config and a shared_state dict,
    return a list of metric instances that are enabled.
    """
    metrics_instances = []

    # Suppose your `cfg.metrics` looks like:
    # metrics:
    #   FID:
    #     enabled: true
    #     some_option: ...
    #   IS:
    #     enabled: false
    #   SSIM:
    #     enabled: true
    #     ...

    # You might do something like:
    if "FID" in cfg.metrics and cfg.metrics.FID.enabled:
        metrics_instances.append(FIDMetric(config=cfg.metrics.FID, shared_state=shared_state))
    # if "IS" in cfg.metrics and cfg.metrics.IS.enabled:
    #     metrics_instances.append(ISMetric(config=cfg.metrics.IS, shared_state=shared_state))
    # if "SSIM" in cfg.metrics and cfg.metrics.SSIM.enabled:
    #     metrics_instances.append(SSIMMetric(config=cfg.metrics.SSIM, shared_state=shared_state))
    # ...

    return metrics_instances


@hydra.main(version_base=None, config_path="../../configs", config_name="metrics")
def main(cfg: DictConfig):
    # shared_state is a dictionary used to cache or share data
    # between metrics. For instance, real image features, etc.
    shared_state = {}

    metrics = instantiate_metrics(cfg, shared_state)

    all_results = {}
    for metric in metrics:
        result = metric.run()  # This calls prepare() then compute()
        all_results.update(result)

    # At this point `all_results` can be complex, with plots etc


if __name__ == "__main__":
    main()
