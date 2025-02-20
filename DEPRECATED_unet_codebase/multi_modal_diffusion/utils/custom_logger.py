import os
import torch as th
import torch.nn as nn

#
# class DebugLogger:
#     def __init__(self, base_dir="debug_logs"):
#         """
#         :param base_dir: The root directory where all debug logs will be stored.
#         """
#         self.base_dir = base_dir
#         os.makedirs(self.base_dir, exist_ok=True)
#         self.current_epoch_log_file = None
#         self.epoch_dir = None
#
#     def start_epoch(self, epoch: int):
#         """
#         Called at the beginning of each epoch to create a new folder + log file.
#         """
#         # Close the previous log file if open:
#         if self.current_epoch_log_file is not None:
#             self.current_epoch_log_file.close()
#
#         # Create an epoch-specific directory:
#         self.epoch_dir = os.path.join(self.base_dir, f"epoch_{epoch}")
#         os.makedirs(self.epoch_dir, exist_ok=True)
#
#         # Open a log file inside that epoch folder:
#         log_file_path = os.path.join(self.epoch_dir, "debug.log")
#         self.current_epoch_log_file = open(log_file_path, "w")
#
#     def log(self, message: str):
#         """
#         Write a message to the current epoch's debug file.
#         """
#         if self.current_epoch_log_file is not None:
#             self.current_epoch_log_file.write(message + "\n")
#             self.current_epoch_log_file.flush()
#         else:
#             # Fallback if someone calls log() without an epoch started
#             print("[DebugLogger WARNING] No epoch started. Printing to stdout:")
#             print(message)
#
#     def close(self):
#         """
#         Clean up. Call this after all training is done if you want to close the file properly.
#         """
#         if self.current_epoch_log_file is not None:
#             self.current_epoch_log_file.close()
#         self.current_epoch_log_file = None
#         self.epoch_dir = None
#
#
# UTILITIES FOR UNET DEBUGGING
def debug_print_stats(tensor: th.Tensor, name: str, debug_logger=None):
    """
    Print shape and basic stats (min, max, mean) for a tensor.
    If debug_logger is provided, log to file. Otherwise, just print to stdout.
    """
    t = tensor.detach()
    msg = (f"{name} - shape: {list(t.shape)}, "
           f"min: {t.min():.4f}, max: {t.max():.4f}, mean: {t.mean():.4f}")
    if debug_logger is not None:
        debug_logger.log(msg)
    else:
        print(msg)


def register_gradient_hooks(module: nn.Module, name_prefix: str = "", debug_logger=None):
    """
        Registers a simple hook on each parameter in the module to print
        gradient stats after backprop.
        If debug_logger is provided, logs gradient stats to file.
        Otherwise, prints to stdout.
    """

    for param_name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        full_name = f"{name_prefix}.{param_name}" if name_prefix else param_name

        def grad_hook(grad, pname=full_name):
            grad_msg = (f"[GRAD HOOK] {pname} -> "
                  f"grad shape={list(grad.shape)}, "
                  f"min={grad.min():.6f}, max={grad.max():.6f}, mean={grad.mean():.6f}")

            if debug_logger is not None:
                debug_logger.log(grad_msg)
            else:
                print(grad_msg)

        param.register_hook(grad_hook)



class DebugLogger:
    def __init__(self, base_dir="debug_logs"):
        """
        :param base_dir: The root directory where all debug logs will be stored.
        """
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.current_epoch_log_file = None
        self.epoch_dir = None

        # For indentation-based hierarchy
        self.indent_level = 0
        self.indent_str = "    "  # 4 spaces per level

    def start_epoch(self, epoch: int):
        """
        Called at the beginning of each epoch to create a new folder + log file.
        """
        if self.current_epoch_log_file is not None:
            self.current_epoch_log_file.close()

        self.epoch_dir = os.path.join(self.base_dir, f"epoch_{epoch}")
        os.makedirs(self.epoch_dir, exist_ok=True)

        log_file_path = os.path.join(self.epoch_dir, "debug.log")
        self.current_epoch_log_file = open(log_file_path, "w")

        # Reset indentation each new epoch, or keep if you prefer
        self.indent_level = 0

    def log(self, message: str):
        """
        Write a message to the current epoch's debug file, with indentation.
        """
        indent = self.indent_str * self.indent_level
        final_msg = indent + message
        if self.current_epoch_log_file is not None:
            self.current_epoch_log_file.write(final_msg + "\n")
            self.current_epoch_log_file.flush()
        else:
            # Fallback
            print("[DebugLogger WARNING] No epoch started. Printing to stdout:")
            print(final_msg)

    def close(self):
        """
        Clean up. Call this after all training is done if you want to close the file properly.
        """
        if self.current_epoch_log_file is not None:
            self.current_epoch_log_file.close()
        self.current_epoch_log_file = None
        self.epoch_dir = None

    def increase_indent(self):
        self.indent_level += 1

    def decrease_indent(self):
        self.indent_level = max(0, self.indent_level - 1)


# wrapper to log
# debug_forward.py

def debug_forward(debug_logger):
    """
    A decorator factory that creates a decorator to log 'Entering/Exiting' messages
    with indentation using the provided `debug_logger`.

    Args:
        debug_logger: An instance of the debug logger to use for logging.

    Returns:
        A decorator that wraps the target method with debug logging.
    """

    def decorator(method):
        def wrapper(self, *args, **kwargs):
            # If no debug, skip indentation logic:
            if not getattr(self, "debug", False):
                return method(self, *args, **kwargs)

            module_name = self.__class__.__name__

            if debug_logger is not None:
                debug_logger.log(f"Entering {module_name}.{method.__name__}()...")
                debug_logger.increase_indent()
            try:
                result = method(self, *args, **kwargs)
            finally:
                if debug_logger is not None:
                    debug_logger.decrease_indent()
                    debug_logger.log(f"Exiting {module_name}.{method.__name__}()...")
            return result

        return wrapper

    return decorator




