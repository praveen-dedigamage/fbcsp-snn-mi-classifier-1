#!/bin/bash
# ============================================================================
#  Roihu setup — run ONCE, interactively, on the GPU login node.
#
#      ssh <user>@roihu-gpu.csc.fi
#      bash roihu/00_setup_roihu.sh
#
#  IMPORTANT — architecture split (CSC docs, "Getting started with Roihu"):
#    Roihu GPU nodes are NVIDIA GH200 Grace Hopper: the CPU side is ARM
#    (aarch64), not x86.  CSC states plainly that software compiled on GPU
#    nodes only works on GPU nodes, "and this also applies to Python
#    environments".  So:
#      * build this environment on roihu-gpu.csc.fi, NOT roihu-cpu.csc.fi
#      * any x86 wheels (including a venv copied from a laptop) will NOT run
#    Verify with:  uname -m   ->  expect aarch64
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== architecture check ==="
ARCH="$(uname -m)"
echo "uname -m = ${ARCH}"
if [[ "${ARCH}" != "aarch64" ]]; then
    echo "!! Expected aarch64 (Roihu GPU login node)."
    echo "!! You appear to be on ${ARCH} — probably roihu-cpu.csc.fi."
    echo "!! Reconnect to roihu-gpu.csc.fi and re-run, or the env will not"
    echo "!! work inside GPU jobs."
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. PyTorch module
# ---------------------------------------------------------------------------
# NOT hard-coded on purpose: the exact module name/version on Roihu must be
# discovered on the system rather than guessed.  Run
#     module spider python-pytorch
# and set PYTORCH_MODULE below to what it reports.
PYTORCH_MODULE="${PYTORCH_MODULE:-python-pytorch/2.10}"

echo
echo "=== available pytorch modules ==="
module spider python-pytorch 2>&1 | head -30 || true
echo
echo "Loading: ${PYTORCH_MODULE}"
module purge
module load "${PYTORCH_MODULE}"

python -c "
import torch, platform
print('machine      :', platform.machine())
print('torch        :', torch.__version__)
print('cuda avail   :', torch.cuda.is_available())
print('device       :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')
"

# ---------------------------------------------------------------------------
# 2. Project Python deps not in the module
# ---------------------------------------------------------------------------
# snntorch / moabb / mne are usually not part of CSC's pytorch module.
# --user installs into ~/.local, which is arch-specific: because this runs on
# the ARM GPU login node, the result is only valid for GPU jobs.  If you also
# need CPU-node runs, build a separate env there.
echo
echo "=== installing project deps (snntorch, moabb, mne) ==="
pip install --user snntorch moabb mne

# ---------------------------------------------------------------------------
# 3. Pre-download the dataset ONCE
# ---------------------------------------------------------------------------
# 60 array tasks starting simultaneously would otherwise all try to download
# BNCI2015-001 into the same ~/mne_data cache and race each other.  Fetch it
# here, serially, before any job is submitted.
echo
echo "=== pre-downloading BNCI2015_001 (all 12 subjects) ==="
python - <<'PY'
import warnings, logging
warnings.filterwarnings("ignore")
for n in ("moabb", "mne"):
    logging.getLogger(n).setLevel(logging.ERROR)
from fbcsp_snn.config import Config
from fbcsp_snn.pipeline import _load_raw

for sid in range(1, 13):
    cfg = Config()
    cfg.source = "moabb"; cfg.moabb_dataset = "BNCI2015_001"
    cfg.subject_id = sid; cfg.n_classes = 2
    Xtr, ytr, Xte, yte = _load_raw(cfg)
    print(f"  S{sid:<3} train {Xtr.shape}  test {Xte.shape}", flush=True)
print("dataset cached under ~/mne_data")
PY

echo
echo "=== setup complete ==="
echo "Next:  sbatch roihu/01_train_array.sh   (edit --account first)"
