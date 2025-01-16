import re
import pandas as pd
import os

def parse_epoch_log(log_lines):
    """
    Parse a full epoch log (with multiple iterations).
    Returns a list of dicts, each containing:
      {
        "iteration": int,               # which iteration this record belongs to
        "region": str,                  # "pre-input", "input", "middle", "output", or None
        "class_path": str,              # e.g. "TimestepEmbedSequential->ResBlock->ImageConv"
        "shape": str,                   # e.g. "4, 192, 64, 64"
        "min": float,
        "max": float,
        "mean": float,
        "raw_line": str                 # original log line (optional)
      }
    """

    # -- REGEX PATTERNS --

    # Lines that contain shape stats:
    # e.g. "shape: [4, 192, 64, 64], min: -3.5586, max: 3.4881, mean: -0.0066"
    shape_stat_pattern = re.compile(
        r"shape:\s*\[(.*?)\],\s*min:\s*([-0-9\.]+),\s*max:\s*([-0-9\.]+),\s*mean:\s*([-0-9\.]+)"
    )

    # "Entering" or "Exiting" something:
    entering_pattern = re.compile(r"Entering\s+(\w+)\.forward\(\)\.*")
    exiting_pattern  = re.compile(r"Exiting\s+(\w+)\.forward\(\)\.*")

    # Region detection patterns:
    #   - "Entering MultimodalUNet.forward()" => region = "pre-input"
    #   - "MultimodalUNet - input_block 0"   => region = "input"
    #   - "MultimodalUNet - middle_blocks"   => region = "middle"
    #   - "MultimodalUNet - output_block 2"  => region = "output"
    re_enter_multimodal = re.compile(r"Entering\s+MultimodalUNet\.forward\(\)\.*")
    re_inp_block        = re.compile(r"MultimodalUNet\s*-\s*input_block\s+(\d+)")
    re_mid_block        = re.compile(r"MultimodalUNet\s*-\s*middle_blocks")
    re_out_block        = re.compile(r"MultimodalUNet\s*-\s*output_block\s+(\d+)")

    # -- STATE VARIABLES --

    # iteration starts at 1; each "[GRAD HOOK]" triggers iteration += 1
    iteration = 1

    # region can be one of "pre-input", "input", "middle", "output", or None
    region = None

    # context stack for nested classes
    context_stack = []

    # list of all parsed entries
    parsed_records = []

    # -- MAIN PARSE LOOP --

    was_hooked = False
    for line in log_lines:
        if "Exiting MultimodalUNet.forward()" in line:
            c = 4
        line = line.rstrip("\n")

        # 0) Detect new iteration boundary:
        #    If line contains "[GRAD HOOK]", we move to the next iteration
        if "[GRAD HOOK]" in line and not was_hooked:
            iteration += 1
            was_hooked = True
            continue

        # 1) Region detection updates:
        if re_enter_multimodal.search(line):
            region = "pre-input"
        else:
            # If we see lines like "MultimodalUNet - input_block 0 output (image)..."
            # or "MultimodalUNet - middle_blocks output..."
            # or "MultimodalUNet - output_block 2..."
            # we set region accordingly.
            inp_match = re_inp_block.search(line)
            if inp_match:
                region = "input"

            mid_match = re_mid_block.search(line)
            if mid_match:
                region = "middle"

            out_match = re_out_block.search(line)
            if out_match:
                region = "output"

        # 2) Check for "Entering X.forward()" -> push on context_stack
        ent = entering_pattern.search(line)
        if ent:
            class_name = ent.group(1)  # e.g. "TimestepEmbedSequential" or "ResBlock"
            context_stack.append(class_name)
            continue

        # 3) Check for "Exiting X.forward()" -> pop from context_stack
        ext = exiting_pattern.search(line)
        if ext:
            class_name = ext.group(1)
            if context_stack and context_stack[-1] == class_name:
                context_stack.pop()
                was_hooked = False
            continue

        # 4) Check for shape/min/max/mean lines
        stat_match = shape_stat_pattern.search(line)
        if stat_match:
            shape_str  = stat_match.group(1)   # e.g. "4, 192, 64, 64"
            min_val    = float(stat_match.group(2))
            max_val    = float(stat_match.group(3))
            mean_val   = float(stat_match.group(4))

            # Build a class_path string from the stack
            if context_stack:
                class_path = "->".join(context_stack)
            else:
                # sometimes we might have no classes in the stack if the line is top-level
                # (like "MultimodalUNet - input_block 0 output (image) - shape: ...")
                class_path = "MultimodalUNet_top"

            record = {
                "iteration": iteration,
                "region": region,
                "class_path": class_path,
                "shape": shape_str,
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "raw_line": line
            }
            parsed_records.append(record)

    return parsed_records


if __name__ == "__main__":

    log_folder_path = "/mnt/storage/nacc_sub/tmp"
    for file_name in os.listdir(log_folder_path):
        # Check if the file has a .log extension
        if file_name.endswith(".log"):
            file_path = os.path.join(log_folder_path, file_name)
            with open(file_path, 'r') as file:
                # Read all lines and strip trailing newlines or spaces
                log_lines = [line for line in file]
                parsed = parse_epoch_log(log_lines)



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