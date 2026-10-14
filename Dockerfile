# AEIC-SE ft2 codec — encoder (CPU, ONNX) + decoder (NVIDIA GPU, diffusion) in one image.
#
# Code + ONNX encoders are baked in. The decoder weights (~4.4 GB) and the
# original/ encoded/ decoded/ folders are mounted at runtime (see docker_run_cmd.md).
#
#   docker build -t aeic-codec .
#
# Python 3.10 is mandatory: the bundled rANS extensions are cpython-310-x86_64 .so files.
# The torch cu121 wheel ships its own CUDA runtime, so a plain Debian base is enough;
# the host only needs an NVIDIA driver (>= 530) and the NVIDIA Container Toolkit.
FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Writable HOME for any uid passed via --user (triton / HF caches land here).
    HOME=/tmp/home

# gcc: triton JIT-compiles a small CUDA helper on first use.
# libgl/libglib: not needed (Pillow only), so they are left out.
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /tmp/home && chmod 1777 /tmp/home

# torch first, pinned to the +cu121 build so a CPU-only wheel can't sneak in.
# pip is upgraded first: the slim image's pip 23.0 rejects the PyTorch index's
# package names ("inconsistent Name"); PyPI stays available for build deps.
RUN pip install pip==24.3.1 \
 && pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
        --index-url https://download.pytorch.org/whl/cu121 \
        --extra-index-url https://pypi.org/simple

# Encoder + decoder runtime deps (versions from environment-minimal.yml;
# benchmark-only metric packages are omitted). pytorch-lightning is not imported by
# the code, but halfDecoder.ckpt pickles Lightning objects, so torch.load needs it.
RUN pip install \
        numpy==1.26.4 \
        pillow==10.4.0 \
        onnxruntime==1.23.2 \
        diffusers==0.25.1 \
        transformers==4.46.3 \
        accelerate==1.9.0 \
        huggingface-hub==0.25.0 \
        peft==0.19.1 \
        einops==0.8.2 \
        scipy==1.11.1 \
        tqdm==4.68.1 \
        xformers==0.0.23.post1 \
        triton==2.1.0 \
        gdown==5.2.0 \
        pytorch-lightning==2.6.5 \
        torchmetrics==1.6.2 \
        lightning-utilities==0.15.3

WORKDIR /app
COPY . /app

# Runtime volumes (empty in the image; see .dockerignore).
RUN mkdir -p /app/original /app/encoded /app/decoded /app/decoder/weights \
 && chmod 777 /app/original /app/encoded /app/decoded /app/decoder/weights

CMD ["bash"]
