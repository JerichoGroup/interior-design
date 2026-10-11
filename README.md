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
./roundtrip.sh /path/to/my_photo.jpg -hd    # smaller/faster variant, see §4
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

## 4. Two canvases: default and `-hd`

The encoder is fixed-shape, so each canvas is its own ONNX file. Both come from the same
checkpoint and share the same entropy tables; **the decoder needs no flag either way** — it
reads the canvas out of the `.bin` header.

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
your own content before switching.

## 5. What is in here

```
encoder/
  run_onnx_npy.py                      encode: neural model + rANS -> .bin   (CPU)
  aeic_entropy.py                      the rANS coder, pure python + numpy
  aeic_ft2_cdf.bin                     entropy tables (0.8 MB, required)
  aeic_se_ft2_encoder_2048x1536.onnx   the encoder, default canvas
  aeic_se_ft2_encoder_1024x768.onnx    the encoder, -hd canvas
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

## 6. Repository notes

The 4.4 GB of decoder weights and every pipeline output are kept out of git (`.gitignore`);
`decoder/weights/`, `encoded/` and `decoded/` ship as empty folders with a `.gitkeep`.

The two ONNX encoders **are** committed, as ordinary git blobs rather than Git LFS objects —
87 MB and 70 MB, both under GitHub's 100 MB per-file hard limit. A fresh clone is therefore
~155 MB and needs no extra tooling. `.gitattributes` documents the one-command migration to
LFS if you ever want it (it rewrites history, so coordinate first).

## 7. Troubleshooting

| Symptom | Cause |
|---|---|
| `ImportError` on `MLCodec_rans` | Not Python 3.10, or not x86_64. Rebuild from `decoder/cpp_src/` — see `decoder/README.md` §6. |
| Decoder prints "no usable CUDA GPU" and falls back | CPU-only torch, or a torch built for a CUDA newer than your driver. Reinstall the cu121 wheel (§1). |
| `ImportError` from `diffusers.models.unet_2d_blocks` | diffusers is not 0.25.1. |
| `ERROR: missing sd-turbo / codec ckpt / vae decoder` | `decoder/weights/` is empty — run §2. |
| `ERROR: shape mismatch on axis 2` | A `.npy` input is used as-is and must already match the canvas. Pass an image instead, or add/drop `-hd`. |
| Encoder killed / OOM | The 2K encode peaks at ~1.6–3.0 GB RAM. Use `-hd` (~4× less). |
| Decoder CUDA OOM | 2K needs ~20 GB VRAM. Use `-hd`, or `--device cpu`. |
