# --- Dockerfile -----------------------------------------------------------
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# 1. Swarm client
COPY swarmlearning-client-py3-none-manylinux_2_24_x86_64.whl /tmp/
RUN pip install --no-cache-dir /tmp/swarmlearning-client-*.whl && rm /tmp/*.whl

# 2. Your deps
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# 3. Your code
WORKDIR /workspace
COPY PRISMS ./PRISMS
ENV PYTHONUNBUFFERED=1
CMD ["python", "PRISMS/trainers/trainer_swarm.py"]
