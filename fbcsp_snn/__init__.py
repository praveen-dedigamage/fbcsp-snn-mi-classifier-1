"""fbcsp_snn package — device detection, logger setup, CUDA config."""

import io
import logging
import sys

import torch
import torch._dynamo

# Suppress Triton/inductor compilation errors globally and fall back to eager.
# Required for Volta GPUs (V100, sm_70) where Triton PTX codegen is broken
# in PyTorch 2.1.x — affects both model forward and loss backward passes.
torch._dynamo.config.suppress_errors = True

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def setup_logger(name: str = "fbcsp_snn", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to stdout.

    Parameters
    ----------
    name : str
        Logger name (default ``"fbcsp_snn"``).
    level : int
        Logging level (default ``logging.INFO``).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    # Force UTF-8 on Windows where the default console encoding is cp1252
    stream: io.TextIOWrapper | logging.StreamHandler
    if hasattr(sys.stdout, "buffer"):
        stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    else:
        stream = sys.stdout
    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
                          datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def _select_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(device)
        _log = setup_logger()
        _log.info("GPU: %s  (%.1f GB VRAM)", props.name, props.total_memory / 1e9)
        # Prefer TF32 on Ampere+ for matmul throughput (no-op on V100/Volta)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Auto-tune cuDNN kernels for fixed input sizes (free speedup per fold)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        setup_logger().info("CUDA not available — running on CPU")
    return device


DEVICE: torch.device = _select_device()
