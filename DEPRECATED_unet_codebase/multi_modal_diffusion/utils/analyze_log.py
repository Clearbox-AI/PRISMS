import re
import pandas as pd
import os
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

def parse_and_merge_hook(rows):
    """
    Parse gradient log rows into a dictionary and merge duplicate keys by averaging values.

    Parameters:
        rows (list of str): List of gradient log strings.

    Returns:
        dict: Dictionary where key -> [average_min, average_max, average_mean].
              Example: {
                "layer.weight": [min_val, max_val, mean_val],
                "layer.bias":   [min_val, max_val, mean_val]
              }
    """
    pattern = r"^(.*?) -> grad.*min=(-?\d[\d.e\-+]*), max=(-?\d[\d.e\-+]*), mean=(-?\d[\d.e\-+]*)"

    aggregated_data = defaultdict(lambda: [0.0, 0.0, 0.0, 0])  # [sum_min, sum_max, sum_mean, count]

    for row in rows:
        row = row.strip()
        match = re.match(pattern, row)
        if match:
            key = ".".join(match.group(1).strip().split(".")[:-1]).split(" ")[-1]
            min_val, max_val, mean_val = map(float, match.groups()[1:])
            aggregated_data[key][0] += min_val
            aggregated_data[key][1] += max_val
            aggregated_data[key][2] += mean_val
            aggregated_data[key][3] += 1

    # Compute averages
    result = {}
    for key, (sum_min, sum_max, sum_mean, count) in aggregated_data.items():
        if count > 0:
            avg_min  = sum_min / count
            avg_max  = sum_max / count
            avg_mean = sum_mean / count
            result[key] = [avg_min, avg_max, avg_mean]
    return result


def parse_epoch_log(log_lines):
    """
    Parse a full epoch log (with multiple iterations).
    Returns a tuple of:
      parsed_records,  # list of dicts with shape stats
      hooks_by_iter    # dict mapping iteration -> parse_and_merge_hook result
    where each entry in parsed_records is:
      {
        "iteration": int,
        "region": str,       # "pre-input", "input", "middle", "output", or None
        "class_path": str,   # e.g. "TimestepEmbedSequential->ResBlock->ImageConv"
        "shape": str,        # e.g. "4, 192, 64, 64"
        "min": float,
        "max": float,
        "mean": float,
        "raw_line": str
      }
    """

    # -- REGEX for shape lines --
    shape_stat_pattern = re.compile(
        r"shape:\s*\[(.*?)\],\s*min:\s*([-0-9\.]+),\s*max:\s*([-0-9\.]+),\s*mean:\s*([-0-9\.]+)"
    )

    # "Entering" or "Exiting" something:
    entering_pattern = re.compile(r"Entering\s+(\w+)\.forward\(\)\.*")
    exiting_pattern  = re.compile(r"Exiting\s+(\w+)\.forward\(\)\.*")

    # Region detection patterns
    re_enter_multimodal = re.compile(r"Entering\s+MultimodalUNet\.forward\(\)\.*")
    re_inp_block        = re.compile(r"MultimodalUNet\s*-\s*input_block\s+(\d+)")
    re_mid_block        = re.compile(r"MultimodalUNet\s*-\s*middle_blocks")
    re_out_block        = re.compile(r"MultimodalUNet\s*-\s*output_block\s+(\d+)")

    iteration = 1
    region = None
    context_stack = []

    parsed_records = []

    # dict of iteration -> list of hook lines
    hook_lines_for_iter = defaultdict(list)

    was_hooked = False
    for line in log_lines:
        # store the raw line for reference
        original_line = line.rstrip("\n")

        # # Check if line is a gradient hook line
        # if "[GRAD HOOK]" in line:
        #     hook_lines_for_iter[iteration].append(original_line)

        # 0) Detect new iteration boundary:
        #    If line contains "[GRAD HOOK]", we move to the next iteration
        if "[GRAD HOOK]" in line:
            if not was_hooked:
                iteration += 1
                was_hooked = True
            hook_lines_for_iter[iteration-1].append(original_line)
            continue

        # region detection
        if re_enter_multimodal.search(original_line):
            region = "pre-input"
        else:
            inp_match = re_inp_block.search(original_line)
            if inp_match:
                region = "input"

            mid_match = re_mid_block.search(original_line)
            if mid_match:
                region = "middle"

            out_match = re_out_block.search(original_line)
            if out_match:
                region = "output"

        # "Entering X.forward()" -> push context
        ent = entering_pattern.search(original_line)
        if ent:
            class_name = ent.group(1)
            context_stack.append(class_name)
            continue

        # "Exiting X.forward()" -> pop context
        ext = exiting_pattern.search(original_line)
        if ext:
            class_name = ext.group(1)
            if context_stack and context_stack[-1] == class_name:
                context_stack.pop()
                was_hooked = False
            continue

        # shape line?
        stat_match = shape_stat_pattern.search(original_line)
        if stat_match:
            shape_str = stat_match.group(1)    # e.g. "4, 192, 64, 64"
            min_val   = float(stat_match.group(2))
            max_val   = float(stat_match.group(3))
            mean_val  = float(stat_match.group(4))

            if context_stack:
                class_path = "->".join(context_stack)
            else:
                class_path = "MultimodalUNet_top"

            record = {
                "iteration": iteration,
                "region": region,
                "class_path": class_path,
                "shape": shape_str,
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "raw_line": original_line,
            }
            parsed_records.append(record)

    # After the loop, parse hook lines for each iteration
    hooks_by_iter = {}
    for iter_i, hook_lines in hook_lines_for_iter.items():
        merged_data = parse_and_merge_hook(hook_lines)
        hooks_by_iter[iter_i] = merged_data

    # Return both shape stats & the merged hook stats by iteration
    return parsed_records, hooks_by_iter


# Example usage
if __name__ == "__main__":

    log_folder_path = "/mnt/storage/nacc_sub/tmp"

    epochs_data = []
    for epoch_number, epoch_folder in enumerate(
            (folder for folder in os.listdir(log_folder_path) if
             Path(log_folder_path, folder).is_dir() and "epoch" in folder), start=1):

        epoch_folder_path = Path(log_folder_path, epoch_folder)
        # e.g. suppose each epoch_folder has exactly one ".log" file
        log_files = [f for f in os.listdir(epoch_folder_path) if f.endswith(".log")]
        if not log_files:
            continue

        log_path = Path(epoch_folder_path, log_files[0])
        with open(log_path, "r") as f:
            log_lines = f.readlines()

        parsed_records, hooks_by_iter = parse_epoch_log(log_lines)
        pd.DataFrame(parsed_records).to_excel(Path(log_folder_path, epoch_folder, "parsed_layers_tensors.xlsx"), index=False)

        gradients_in_epoch = {
            key: [
                sum(values) / len(values) for values in zip(*[sub[key] for sub in hooks_by_iter.values()])
            ] for key in next(iter(hooks_by_iter.values()))
        }
        epochs_data.append((epoch_number, gradients_in_epoch))

    # Sort epochs_data by the epoch_number just in case the folder listing is out of order
    epochs_data.sort(key=lambda x: x[0])

    # Collect data for plotting
    param_names = list(epochs_data[0][1].keys())
    metrics = ["avg_min", "avg_max", "avg_mean"]
    epochs = [epoch_data[0] for epoch_data in epochs_data]

    data_by_metric = {metric: {param: [] for param in param_names} for metric in metrics}

    for epoch_number, gradients in epochs_data:
        for param_name, values in gradients.items():
            for idx, metric in enumerate(metrics):
                data_by_metric[metric][param_name].append(values[idx])

    # ------------------------------------------------------------
    # Create a folder for saving plots
    # ------------------------------------------------------------

    output_plots_path = Path(log_folder_path, "plots")
    output_plots_path.mkdir(exist_ok=True, parents=True)

    # Plot each metric, but use a log scale on the y-axis
    for metric, params_data in data_by_metric.items():
        plt.figure(figsize=(24, 16))
        for param_name, values in params_data.items():
            plt.plot(epochs, values, label=param_name)

        plt.title(f"Gradient {metric} Across Epochs", fontsize=14)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel(f"{metric}", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(fontsize=8, title="Param Names", title_fontsize=10)
        plt.tight_layout()
        plt.grid(True)

        # --------------------------
        # Make y-axis logarithmic
        # --------------------------
        plt.yscale("log")

        # Optional: set y-limits to focus on small gradients (tweak as needed)
        # For example, show from 1e-8 up to 1e-2:
        if metric == "avg_max":
            plt.ylim(1e-6, 1e-1)
        elif metric == "avg_min":
            # plt.ylim(-1e-9, -1e-3)
            ...
        elif metric == "avg_mean":
            plt.ylim(1e-12, 1e-5)

        # Save the plot
        plt.savefig(Path(output_plots_path, f"gradients_{metric}_png"))
        plt.close()

    # df = pd.DataFrame(parsed)
    #
    # print("\n--- Parsed DataFrame ---")
    # print(df)
    #
    # # Example grouping: get min-of-min, max-of-max, average of means by iteration & region:
    # grouped = df.groupby(["iteration", "region"]).agg({
    #     "min": "min",
    #     "max": "max",
    #     "mean": "mean"
    # })
    # print("\n--- Aggregated Stats (by iteration & region) ---")
    # print(grouped)
    #
    # # If you want specifically "ResBlock" stats:
    # df_resblock = df[df["class_path"].str.contains("ResBlock")]
    # print("\n--- All ResBlock logs ---")
    # print(df_resblock)

# iteration, region, class_path, shape, min, max, mean, raw_line
# 1,pre-input,MultimodalUNet,"4, 4, 64, 64",-4.2022,4.9003,-0.0751,"MultimodalUNet.forward - image (input) - shape: [4, 4, 64, 64], min: -4.2022, max: 4.9003, mean: -0.0751"
# 1,pre-input,MultimodalUNet,"4, 174",-3.2299,2.8158,-0.0659,"MultimodalUNet.forward - tabular (input) - shape: [4, 174], min: -3.2299, max: 2.8158, mean: -0.0659"