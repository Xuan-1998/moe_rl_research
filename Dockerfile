FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    ninja-build \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    bitsandbytes \
    gymnasium \
    stable-baselines3 \
    pybind11 \
    packaging \
    ninja

WORKDIR /workspace/moe_rl_research

CMD ["/bin/bash"]
