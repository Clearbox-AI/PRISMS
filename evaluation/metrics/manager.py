from typing import Dict, Any, List

from data.artifact_store import ArtifactStore
from data.data_bucket import DataBucket
from enums.metrics import MetricRequirementType, MetricType
from data.data_operations import load_generated_data, load_data_from_config, get_model_for_metric
from utils.configurations import load_hydra_config

class MetricManager:
    """
    Responsible for:
      1) Identifying the requirements for a given metric (FID, Accuracy, etc.).
      2) Fetching/creating the required data (real, generated, model, etc.).
      3) Calling the appropriate function to compute the metric.
    """

    def __init__(self):
        """
        Construct a DAG for each metric in a dictionary:
          self.dag_map[MetricType] = { <requirement>: [list_of_dependency_requirements], ... }
        """
        self.dag_map = {
            MetricType.FID: {
                # REAL_DATA has no dependencies
                MetricRequirementType.REAL_DATA: [],
                # GENERATED_DATA depends on REAL_DATA
                MetricRequirementType.GENERATED_DATA: [MetricRequirementType.REAL_DATA],
            },
            MetricType.Other: {
                # For example, let's say other needs REAL_DATA and GROUND_TRUTH_LABELS first
                MetricRequirementType.REAL_DATA: [],
                MetricRequirementType.GROUND_TRUTH_LABELS: [MetricRequirementType.REAL_DATA],
                MetricRequirementType.ML_MODEL: [MetricRequirementType.REAL_DATA, MetricRequirementType.GROUND_TRUTH_LABELS],
                # or any other structure needed
            }
        }

    def compute_metric(
        self,
        store: "ArtifactStore",
        metric_type: MetricType,
        config: Dict[str, Any],
        force: bool = False
    ) -> float:
        """
        Main entry point for computing a metric. We:
          1) Build or retrieve the DAG for the chosen metric_type.
          2) Topologically sort it.
          3) For each requirement in the sorted list, fetch/construct the resource.
          4) Then call the appropriate compute function.
        """
        if metric_type not in self.dag_map:
            raise ValueError(f"No DAG found for metric type {metric_type}")

        dag = self.dag_map[metric_type]

        # Topologically sort the DAG
        sorted_requirements = self._topological_sort(dag)

        # Now fetch each requirement
        resources = {}
        for req_type in sorted_requirements:
            self._fetch_requirement(
                store=store,
                req_type=req_type,
                config=config,
                resources=resources,
                force=force
            )

        # Dispatch to the correct compute method
        if metric_type == MetricType.FID:
            return self._compute_fid(
                real_data=resources[MetricRequirementType.REAL_DATA],
                generated_data=resources[MetricRequirementType.GENERATED_DATA]
            )
        elif metric_type == MetricType.Other:
            return self._compute_other(
                real_data=resources[MetricRequirementType.REAL_DATA],
                ground_truth=resources[MetricRequirementType.GROUND_TRUTH_LABELS],
                model=resources[MetricRequirementType.ML_MODEL]
            )
        else:
            raise ValueError(f"Unsupported metric: {metric_type}")


    def _topological_sort(self, dag: Dict[MetricRequirementType, List[MetricRequirementType]]) -> List[MetricRequirementType]:
        """
        Standard topological sort for a DAG given as an adjacency list:
        dag[node] = [list_of_dependencies].
        That means we must process each item in 'dag[node]' before 'node'.

        The result is a list of nodes in dependency order: each node appears
        after all of its dependencies.
        """
        # We'll do a DFS-based topological sort or a standard in-degree approach.
        # Here is an in-degree approach:
        in_degree_map = {node: 0 for node in dag}
        for node, deps in dag.items():
            for dep in deps:
                in_degree_map[node] = in_degree_map[node] + 1

        # Initialize a queue with all nodes that have in-degree=0
        queue = [n for n, deg in in_degree_map.items() if deg == 0]
        topo_order = []

        while queue:
            current = queue.pop()
            topo_order.append(current)
            # Now reduce in-degree of all nodes that depend on 'current'
            # i.e., find all nodes for which 'current' is a dependency
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
        force: bool = False
    ) -> None:
        """
        Retrieve or create the resource for a single requirement.
        We can read from 'resources' to get any already-fetched dependencies.
        Then we store the final object in resources[req_type].
        """
        if req_type == MetricRequirementType.REAL_DATA:
            real_cfg = config["real_data"]
            resources[MetricRequirementType.REAL_DATA] = load_data_from_config(store, real_cfg, force=force)

        elif req_type == MetricRequirementType.GENERATED_DATA:
            gen_cfg = config["gen_data"]
            # Possibly read "condition_bucket" from real data if needed
            cond_bucket = resources.get(MetricRequirementType.REAL_DATA)
            resources[MetricRequirementType.GENERATED_DATA] = load_generated_data(
                store=store,
                condition_bucket=cond_bucket,
                cfg=gen_cfg,
                force=force
            )

        elif req_type == MetricRequirementType.GROUND_TRUTH_LABELS:
            # TODO: the following is an example, to complete for future metrics
            real_data_bucket = resources[MetricRequirementType.REAL_DATA]
            if real_data_bucket.metadata and "labels" in real_data_bucket.metadata:
                resources[MetricRequirementType.GROUND_TRUTH_LABELS] = real_data_bucket.metadata["labels"]
            else:
                raise NotImplementedError("No ground truth found in metadata or config.")

        elif req_type == MetricRequirementType.ML_MODEL:
            model_cfg = config["model"]
            # The model might rely on real_data or generated_data (depending on your pipeline)
            # Example: Let's say we train on real_data
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


    def _compute_FID(self, real_data: "DataBucket", generated_data: "DataBucket") -> float:
        print("[_compute_fid] Called with real_data and generated_data.")
        return ...

    def _compute_other(
        self,
        real_data: "DataBucket",
        ground_truth: Any,
        model: Any
    ) -> float:
        print("[_compute_other] Called with real_data, ground_truth, model.")
        return ...

if __name__ == "__main__":
    from data.artifact_store import ArtifactStore

    store = ArtifactStore(artifact_root="/mnt/dataset_storage/artifact_store")
    manager = MetricManager()

    fid_config = load_hydra_config("metrics", "FID_nacc_tab_cond_gen")

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