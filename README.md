# AEIC-SE ft2 — neural image codec, encoder + decoder

Compresses a photo to a **3–13 KB** file (≈0.03 bpp, a few hundred times smaller than the
source) and reconstructs it with a one-step diffusion decoder.

The codec is split in two halves that normally run on different machines:

| | where it runs | needs | speed |
|---|---|---|---|
| **encoder** | phone / any CPU box | `onnxruntime` + `numpy` | ~2 s per 2K image |
| **decoder** | desktop with an NVIDIA GPU | torch + diffusers, ~20 GB VRAM for 2K | ~17 s per 2K image |

Both are in this repo so you can run the round trip on one machine.

```
original/  ->  encoder  ->  encoded/*.bin (+ .json)  ->  decoder  ->  decoded/*.png
```

---

## 1. Install (once)

```bash
conda env create -f environment-minimal.yml     # creates the env `aeic-min`
conda activate aeic-min
```

Three constraints are not negotiable, and the env file pins them:
**Python 3.10** (the bundled rANS extension is a prebuilt `cpython-310-x86_64` `.so`),
**diffusers 0.25.1** (the decoder imports private module paths that moved in later versions),
and **torch 2.1.2 built for CUDA 12.1**. If `torch.cuda.is_available()` is False, you got a
CPU wheel — reinstall with:

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

`environment.yml` is the full, exact export of the environment this was developed in; use it
instead if you want a faithful clone rather than the trimmed dependency set.

## 2. Get the decoder weights (once, ~4.4 GB)

They are **not** in the repo. Fetch them:

```bash
cd decoder && ./setup.sh && cd ..
```

`setup.sh` downloads `sd-turbo` (public, on Hugging Face) automatically. The two project
checkpoints — `AEIC_SE_ft2.pkl` (437 MB) and `halfDecoder.ckpt` (361 MB) — are not public:
copy them into `decoder/weights/` by hand, or put their Google Drive file IDs at the top of
`setup.sh` and re-run it. When you are done, `decoder/weights/` must contain:

```
decoder/weights/
  sd-turbo/            (model_index.json, vae/, unet/)
  AEIC_SE_ft2.pkl
  halfDecoder.ckpt
```

The **encoder needs none of this** — its model is in the repo. If you only want to encode,
skip this step entirely.

## 3. Run it

The one-liner, encode and decode together:

```bash
./roundtrip.sh input_2048x1536.png          # the bundled sample
./roundtrip.sh /path/to/my_photo.jpg        # any image, any size
./roundtrip.sh /path/to/my_photo.jpg -hd    # smaller/faster variant (also -fhd, -sd), see §5
```

Or the two steps by hand, which is what you would actually deploy:

```bash
# --- encode: any image -> encoded/myshot.bin (+ myshot.json). CPU only.
cd encoder
python run_onnx_npy.py -i /path/to/my_photo.jpg -o myshot --provider cpu

# --- decode: encoded/myshot.bin -> decoded/myshot.png. NVIDIA GPU.
cd ../decoder
python decode.py myshot.bin
python decode.py myshot.bin --device cpu     # no GPU: works, but minutes per image
```

`-i` and the decoder's `.bin` argument take either a full path or a bare filename, which is
looked up in `original/` and `encoded/` respectively. `-o` names the output; omit it and the
input's name is reused.

Any input size and aspect ratio is accepted. The encoder EXIF-uprights the image, rotates
portrait to landscape, and resizes it to the model's fixed canvas, recording the original
size and orientation in the `.json` sidecar so the decoder restores them. The sidecar is
optional on decode — without it you get the raw canvas, uncropped and unrotated.

## 4. Flags

### Encoder — `encoder/run_onnx_npy.py`

| Flag | Default | What it does |
|------|---------|--------------|
| `-i`, `--input` (alias `--npy`) | `original/input_2048x1536.png` | Input image (`.png/.jpg/…`) or preprocessed `.npy`. A path, or a bare filename looked up in `original/`. Images are EXIF-uprighted, portrait is rotated to landscape (restored on decode), and resized to the chosen size. A `.npy` is used as-is, so it must already match the model's input shape. |
| `-o`, `--output` | input's name | Output base name → `encoded/<name>.bin` + `<name>.json` sidecar. |
| `-fhd`, `--fhd` | off | Encode at **1440×1080** (model `aeic_se_ft2_encoder_1472x1088.onnx`). |
| `-hd`, `--hd` | off | Encode at **1024×768** (model `aeic_se_ft2_encoder_1024x768.onnx`). |
| `-sd`, `--sd` | off | Encode at **640×480** (model `aeic_se_ft2_encoder_640x512.onnx`). |
| *(none of the three)* | — | Encode at **2048×1536** (model `aeic_se_ft2_encoder_2048x1536.onnx`). |
| `--provider {auto,cpu,gpu}` | `auto` | onnxruntime execution provider. `cpu` forces CPU; `gpu` uses CUDA/other if installed. |
| `--model PATH` | per size flag | Use a specific `.onnx` instead. Its input shape overrides the size flag. |
| `--encoded-dir DIR` | `encoded/` | Where the `.bin`/`.json` are written. |
| `--cdf PATH` | `encoder/aeic_ft2_cdf.bin` | Entropy tables for the rANS step. |
| `--no-encode` | off | Run only the neural model (RAM/compatibility check); no `.bin` is written. |

`-fhd`, `-hd` and `-sd` are mutually exclusive.

### Decoder — `decoder/decode.py`

| Flag | Default | What it does |
|------|---------|--------------|
| `bin` (positional) | — | The `.bin` to decode: a path, or a bare name looked up in `encoded/`. |
| `-o`, `--output` | `.bin`'s name | Output PNG: a path, or a bare name → `decoded/<name>.png`. |
| `--decoded-dir DIR` | `decoded/` | Where the PNG is written. |
| `--sidecar PATH` | `<bin>.json` next to the `.bin` | Sidecar with original size/orientation. Without one, the full padded canvas is decoded (no crop/rotate/upscale). |
| `--device {auto,cuda,cpu}` | `auto` | `auto` = GPU if available. `cpu` works without a GPU but takes minutes per 2K image. |
| `--sd-path`, `--codec-ckpt`, `--vae-decoder` | `decoder/weights/` | Override weight locations (also via `AEIC_SD_PATH`, `AEIC_CODEC_CKPT`, `AEIC_VAE_DECODER`). |

The decoder has **no size flag**: it reads the canvas from the `.bin` header, crops any
padding, and upscales back to the original resolution from the sidecar.

### `roundtrip.sh`

`./roundtrip.sh IMAGE [-fhd|-hd|-sd]` runs both steps. The size flag is passed to the encoder,
and the output is named `<image>` or `<image>_<size>` (e.g. `photo_fhd.bin`/`.png`). Set
`PYTHON=/path/to/python` to choose the interpreter.

## 5. Canvases: default, `-fhd`, `-hd`, `-sd`

The encoder is fixed-shape, so each canvas is its own ONNX file. All come from the same
checkpoint and share the same entropy tables; **the decoder needs no flag either way** — it
reads the canvas out of the `.bin` header.

Measured on the bundled `original/input_2048x1536.png`, PSNR after the decoder upscales back
to 2048×1536:

| | default | `-fhd` | `-hd` | `-sd` |
|---|---|---|---|---|
| Image size | 2048×1536 | 1440×1080 | 1024×768 | 640×480 |
| Model canvas | 2048×1536 | 1472×1088 | 1024×768 | 640×512 |
| Model size | 87 MB | 76 MB | 70 MB | 67 MB |
| `.bin` | 10 840 B (0.0276 bpp) | 7 104 B (0.0181 bpp) | 4 306 B (0.0110 bpp) | 2 261 B (0.0058 bpp) |
| PSNR | 30.10 dB | 28.94 dB | 26.65 dB | 23.61 dB |

The decoder only accepts canvas sides that are multiples of 64, and 1080 and 480 aren't. So
`-fhd` and `-sd` resize the image to 1440×1080 / 640×480, fill the extra right/bottom strip by
repeating the edge pixels up to the model canvas, and record the real size in the sidecar so
the decoder crops the strip off.

`-hd` vs default on 100 DIV2K images:

| | default | `-hd` |
|---|---|---|
| Canvas | 2048×1536 | 1024×768 |
| Model | `aeic_se_ft2_encoder_2048x1536.onnx` (87 MB) | `aeic_se_ft2_encoder_1024x768.onnx` (70 MB) |
| Mean size, 100 DIV2K images | **13.07 KB** | **3.80 KB** |
| Max size over those 100 | 20.95 KB | 6.22 KB |
| Mean PSNR / LPIPS | 23.39 dB / 0.142 | 21.14 dB / 0.331 |
| Decode time, same GPU | 16.7 s | 0.87 s |
| Decoder peak VRAM | ~20 GB | far lower |

So `-hd` buys a 3.4× smaller file and a ~19× faster decode for about **−2.25 dB** on natural
photos. Be careful with UI screenshots: there the penalty measured **−6.4 to −7.4 dB**,
because their content is small high-contrast text and a 2× downscale destroys it. Judge it on
your own content before switching. `-fhd` and `-sd` haven't been run on DIV2K yet; expect
the same pattern, with `-fhd` between default and `-hd` and `-sd` below `-hd`.

## 6. What is in here

```
encoder/
  run_onnx_npy.py                      encode: neural model + rANS -> .bin   (CPU)
  aeic_entropy.py                      the rANS coder, pure python + numpy
  aeic_ft2_cdf.bin                     entropy tables (0.8 MB, required)
  aeic_se_ft2_encoder_2048x1536.onnx   the encoder, default canvas
  aeic_se_ft2_encoder_1472x1088.onnx   the encoder, -fhd canvas
  aeic_se_ft2_encoder_1024x768.onnx    the encoder, -hd canvas
  aeic_se_ft2_encoder_640x512.onnx     the encoder, -sd canvas
decoder/
  decode.py                            decode: rANS -> diffusion -> VAE -> PNG (GPU)
  aeic/                                the codec/model source the decoder imports
  cpp_src/                             rANS C++ sources, only needed to rebuild the .so
  sample/                              two ready-to-decode .bin/.json pairs
  setup.sh                             fetches weights/ (not in the repo)
  weights/                             4.4 GB of model weights land here (gitignored)
original/   input images                encoded/   .bin output      decoded/   .png output
environment.yml / environment-minimal.yml
roundtrip.sh                           encode + decode in one command
```

Each half has its own `README.md` with the full detail — input tensor contract, bitstream
layout, memory figures, and how to rebuild the rANS extension on another platform.

## 7. Repository notes

The 4.4 GB of decoder weights and every pipeline output are kept out of git (`.gitignore`);
`decoder/weights/`, `encoded/` and `decoded/` ship as empty folders with a `.gitkeep`.

The four ONNX encoders **are** committed, as ordinary git blobs rather than Git LFS objects —
87, 76, 70 and 67 MB, all under GitHub's 100 MB per-file hard limit. A fresh clone is therefore
~300 MB and needs no extra tooling. `.gitattributes` documents the one-command migration to
LFS if you ever want it (it rewrites history, so coordinate first).

## 8. Troubleshooting

| Symptom | Cause |
|---|---|
| `ImportError` on `MLCodec_rans` | Not Python 3.10, or not x86_64. Rebuild from `decoder/cpp_src/` — see `decoder/README.md` §6. |
| Decoder prints "no usable CUDA GPU" and falls back | CPU-only torch, or a torch built for a CUDA newer than your driver. Reinstall the cu121 wheel (§1). |
| `ImportError` from `diffusers.models.unet_2d_blocks` | diffusers is not 0.25.1. |
| `ERROR: missing sd-turbo / codec ckpt / vae decoder` | `decoder/weights/` is empty — run §2. |
| `ERROR: shape mismatch on axis 2` | A `.npy` input is used as-is and must already match the canvas. Pass an image instead, or pick the matching `-fhd`/`-hd`/`-sd` flag. |
| Encoder killed / OOM | The 2K encode peaks at ~1.6–3.0 GB RAM. Use `-fhd`, `-hd` (~4× less) or `-sd`. |
| Decoder CUDA OOM | 2K needs ~20 GB VRAM. Use a smaller size flag, or `--device cpu`. |
