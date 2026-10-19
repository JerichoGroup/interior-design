# Running the AEIC codec in Docker

Code and ONNX encoders are inside the image. Mounted from the host:

| host                     | container              | used by          |
|--------------------------|------------------------|------------------|
| `$(pwd)/original`        | `/app/original`        | encoder (input)  |
| `$(pwd)/encoded`         | `/app/encoded`         | encoder → decoder|
| `$(pwd)/decoded`         | `/app/decoded`         | decoder (output) |
| `$(pwd)/decoder/weights` | `/app/decoder/weights` | decoder (read-only) |

Prerequisites: NVIDIA driver + NVIDIA Container Toolkit on the host, and
`decoder/weights/` populated (`cd decoder && ./setup.sh`). Run everything from the repo root.

## Build

```bash
docker build -t aeic-codec .
```

## Round trip (encode + decode)

Put the image in `./original/` and pass the bare filename:

```bash
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$(pwd)/original:/app/original" \
  -v "$(pwd)/encoded:/app/encoded" \
  -v "$(pwd)/decoded:/app/decoded" \
  -v "$(pwd)/decoder/weights:/app/decoder/weights:ro" \
  aeic-codec ./roundtrip.sh input_2048x1536.png        # add -fhd / -hd / -sd for a smaller canvas
```

## Encode only (CPU, no GPU or weights needed)

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$(pwd)/original:/app/original" \
  -v "$(pwd)/encoded:/app/encoded" \
  -w /app/encoder \
  aeic-codec python run_onnx_npy.py -i input_2048x1536.png -o myshot --provider cpu
```

## Decode only (GPU)

```bash
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$(pwd)/encoded:/app/encoded" \
  -v "$(pwd)/decoded:/app/decoded" \
  -v "$(pwd)/decoder/weights:/app/decoder/weights:ro" \
  -w /app/decoder \
  aeic-codec python decode.py myshot.bin              # append --device cpu if no GPU
```

## Interactive shell

```bash
docker run --rm -it --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$(pwd)/original:/app/original" \
  -v "$(pwd)/encoded:/app/encoded" \
  -v "$(pwd)/decoded:/app/decoded" \
  -v "$(pwd)/decoder/weights:/app/decoder/weights:ro" \
  aeic-codec
```

Notes:
- `--user` keeps output files owned by you instead of root.
- 2K decode needs ~20 GB VRAM; pick a GPU with `--gpus '"device=0"'`.
