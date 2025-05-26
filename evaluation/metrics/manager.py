from typing import Dict, Any, List

from enums.metrics import MetricRequirementType, MetricType
from evaluation.metrics.base_metric import BaseMetric
from data.data_operations import load_data_from_config, load_generated_data, get_model_for_metric
from data.artifact_store import ArtifactStore
from evaluation.metrics.FID import FidMetric
from evaluation.metrics.other import OtherMetric
from hydra import compose, initialize_config_dir
import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

class MetricManager:
    def __init__(self, metrics: List[BaseMetric]):
        """
        Initialize with a list of available metric classes/instances.
        We'll store them in a dict keyed by metric_type.
        """
        self.metric_map = {}
        for metric in metrics:
            self.metric_map[metric.metric_type] = metric

    def compute_metric(
        self,
        store: "ArtifactStore",
        metric_type: MetricType,
        config: DictConfig,
        force: bool = False
    ) -> float:
        """
        Given a metric_type, we find the corresponding metric object,
        topologically sort its DAG, fetch the requirements, then call its compute().
        """
        metric_obj = self.metric_map.get(metric_type)
        if metric_obj is None:
            raise ValueError(f"No metric defined for metric_type {metric_type}")

        # 1) Topologically sort this metric's DAG
        dag = metric_obj.dag
        sorted_requirements = self._topological_sort(dag)

        # 2) Fetch each requirement in that order
        resources: Dict[MetricRequirementType, Any] = {}
        for req_type in sorted_requirements:
            self._fetch_requirement(
                store=store,
                req_type=req_type,
                config=config,
                resources=resources,
                force=force
            )

        # 3) Finally, compute the metric
        return metric_obj.compute(resources)

    def _topological_sort(self, dag: Dict[MetricRequirementType, List[MetricRequirementType]]) -> List[MetricRequirementType]:
        """
        Standard in-degree approach.
        dag[node] = [list_of_dependencies].
        """
        in_degree_map = {node: 0 for node in dag}
        for node, deps in dag.items():
            for dep in deps:
                in_degree_map[node] += 1

        queue = [n for n, deg in in_degree_map.items() if deg == 0]
        topo_order = []

        while queue:
            current = queue.pop()
            topo_order.append(current)
            # reduce in-degree of all nodes that depend on 'current'
            for node, deps in dag.items():
                if current in deps:
                    in_degree_map[node] -= 1
                    if in_degree_map[node] == 0:
                        queue.append(node)

        if len(topo_order) != len(dag):
            raise ValueError("Cycle detected or the DAG is incomplete.")
        return topo_order

    def _fetch_requirement(
        self,
        store: "ArtifactStore",
        req_type: MetricRequirementType,
        config: Dict[str, Any],
        resources: Dict[MetricRequirementType, Any],
        force: bool
    ):
        """
        Retrieve or create the resource for a single requirement,
        storing it in resources[req_type].
        """

        if req_type == MetricRequirementType.REAL_DATA:
            real_cfg = config["real_data"]
            resources[MetricRequirementType.REAL_DATA] = load_data_from_config(store, real_cfg, force=force)

        elif req_type == MetricRequirementType.GENERATED_DATA:
            gen_cfg = config["gen_data"]
            cond_bucket = resources.get(MetricRequirementType.REAL_DATA)
            resources[MetricRequirementType.GENERATED_DATA] = load_generated_data(
                store=store,
                condition_bucket=cond_bucket,
                cfg=gen_cfg,
                force=force
            )

        elif req_type == MetricRequirementType.GROUND_TRUTH_LABELS:
            real_data_bucket = resources[MetricRequirementType.REAL_DATA]
            if real_data_bucket.metadata and "labels" in real_data_bucket.metadata:
                resources[MetricRequirementType.GROUND_TRUTH_LABELS] = real_data_bucket.metadata["labels"]
            else:
                raise NotImplementedError("No ground truth found in metadata or config.")

        elif req_type == MetricRequirementType.ML_MODEL:
            model_cfg = config["model"]
            training_data = resources.get(MetricRequirementType.REAL_DATA)
            model_obj = get_model_for_metric(
                store=store,
                model_cfg=model_cfg,
                training_data=training_data,
                force=force
            )
            resources[MetricRequirementType.ML_MODEL] = model_obj

        else:
            raise ValueError(f"Unknown requirement type: {req_type}")

if __name__ == "__main__":

    fid_metric = FidMetric()
    other_metric = OtherMetric()
    manager = MetricManager(metrics=[fid_metric, other_metric])

    store = ArtifactStore(artifact_root="/mnt/dataset_storage/artifact_store")

    from utils.configurations import set_project_root
    set_project_root()
    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "metrics"))):
        fid_config = compose(config_name="FID_nacc_tab_cond_gen")
        OmegaConf.set_struct(fid_config, False)

    fid_value = manager.compute_metric(
        store=store,
        metric_type=MetricType.FID,
        config=fid_config
    )
    print("Computed FID:", fid_value)

    # Example config for Other
    o_config = {
        "real_data": {
            "artifact_key": "my_training_data",
            "source_type": "folder",
            "data_label": "tab",
            "folder_path": "/path/to/training_dataset",
            # Possibly includes metadata with 'labels'
        },
        "model": {
            "artifact_key": "my_classifier",
            "load_checkpoint_path": "/path/to/ckpt.pth",
            # or "train_params": {...} if it needs training
        },
        # ground_truth logic could be in real_data.metadata or a separate config
    }

    o_value = manager.compute_metric(
        store=store,
        metric_type=MetricType.Other,
        config=o_config
    )
    print("Computed Other:", o_value)