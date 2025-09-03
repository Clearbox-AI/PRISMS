# --- Dockerfile ------------------------------------------------------------
# 1) Lean Python 3.9 base so we control the interpreter version
FROM python:3.9-slim

# 2) System deps many ML wheels expect (build tools kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential curl ca-certificates \
      libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 3) Pin protobuf to a version that works with Swarm’s generated _pb2 files
#    (TF 2.18 needs >=3.20.3; Swarm needs 3.20.x; the common sweet spot is 3.20.3)
RUN python -m pip install --upgrade pip \
 && python -m pip install --no-cache-dir "protobuf==3.20.3"

# 4) PyTorch + CUDA 11.8 (wheels include CUDA user libs)
RUN python -m pip install --no-cache-dir \
      torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu118

# 5) Copy your HPE Swarm wheel (we’ll fix its filename, then install)
COPY PRISMS/swarmlearning-client-py3-none-manylinux_2_24_x86_64.whl /tmp/sl_badname.whl

# 5a) Fix the wheel filename using its own METADATA/WHEEL (write path to /tmp/sl_fixed.txt)
RUN set -eux; \
  python - <<'PY' >/tmp/sl_fixed.txt
import zipfile, re, os
src = "/tmp/sl_badname.whl"
with zipfile.ZipFile(src) as z:
    meta_name = [n for n in z.namelist() if n.endswith("METADATA")][0]
    wheel_name = [n for n in z.namelist() if n.endswith("WHEEL")][0]
    meta = z.read(meta_name).decode("utf-8", "replace")
    wheel = z.read(wheel_name).decode("utf-8", "replace")
name = re.search(r"^Name:\s*(.+)$", meta, re.M).group(1).strip().replace('-', '_')
version = re.search(r"^Version:\s*([^\s]+)$", meta, re.M).group(1).strip()
tag = re.search(r"^Tag:\s*([^\s]+)$", wheel, re.M).group(1).strip()  # e.g. py3-none-manylinux_2_24_x86_64
py, abi, plat = tag.split('-')
fixed = f"/tmp/{name}-{version}-{py}-{abi}-{plat}.whl"
os.rename(src, fixed)
print(fixed)
PY

# 5b) Install the fixed wheel (no deps so pip doesn’t fight protobuf)
RUN set -eux; \
  SL_WHL="$(cat /tmp/sl_fixed.txt)"; \
  python -m pip install --no-cache-dir --no-deps "$SL_WHL"; \
  rm -f "$SL_WHL" /tmp/sl_fixed.txt

# 6) Your Python deps (avoid re-installing torch/vision/audio here).
#    We also cap protobuf via a constraints file so nothing upgrades it.
COPY PRISMS/requirements.txt /tmp/requirements.txt
RUN printf "protobuf==3.20.3\n" > /tmp/constraints.txt \
 && python -m pip install --no-cache-dir -r /tmp/requirements.txt -c /tmp/constraints.txt \
 && rm -f /tmp/requirements.txt /tmp/constraints.txt

# 7) Your code
WORKDIR /workspace
COPY PRISMS /workspace/PRISMS

ENV PYTHONUNBUFFERED=1
CMD ["python", "PRISMS/trainers/trainer_swarm.py"]