# AEIC-SE ft2 — home-side decoder (standalone)

Give this folder's script an encoded `.bin` and it reconstructs the image. This is the
**heavy half** of the codec and runs on a **home PC with an NVIDIA GPU** — the phone produces
the `.bin`, this turns it back into a PNG.

This folder is **self-contained**: the decoder code, the AEIC source it needs, the C++ rANS
`.so` files, and **all model weights** are already inside. The only thing you set up on the
PC is a Python/CUDA environment (the weights are bundled, but torch/CUDA still must be installed).

---

## 0. The big picture

- The phone **encodes** a photo into a tiny `.bin` (~10–20 KB) plus a small `.json` sidecar
  (original size / orientation). It cannot display it — the `.bin` is a neural latent.
- This decoder does the rest: **rANS entropy-decode → 1-step diffusion (SD-Turbo) → tiled VAE
  → crop to content → rotate back → save a ~2K PNG.**

---

## 1. Requirements

- **OS / CPU:** Linux, **x86_64**.
- **GPU (default):** an **NVIDIA CUDA** GPU. A 2K image peaks at **~17–20 GB VRAM**, so a
  **24 GB** card (RTX 3090/4090, Quadro RTX 6000, …) is comfortable.
- **No GPU? use `--device cpu`** — runs the whole decode on the CPU (needs lots of RAM, and is
  **slow**: several minutes for a 2K image). Good for a one-off check on a GPU-less PC.
- **Python:** 3.10 (the bundled rANS `.so` files are built for CPython 3.10 + x86_64 — see §6).

---

## 2. One-time setup (env only — weights are already here)

```bash
cd decoder
conda create -n aeic python=3.10 -y && conda activate aeic
pip install -r requirements.txt
```

That's it. The weights are bundled under `weights/` (sd-turbo vae+unet, `AEIC_SE_ft2.pkl`,
`halfDecoder.ckpt`) — **no download step**. (`setup.sh` exists only to re-fetch them if you
ever delete them; you don't need to run it.)

---

## 3. Decode a `.bin`

> **Run it in the `aeic` env** (`conda activate aeic`). A bare `python` may point at another
> env whose torch can't see your GPU — see Troubleshooting.

Inside the repo layout the `.bin` is read from `../encoded/` and the PNG is
written to `../decoded/` by default:

```bash
python decode.py myshot.bin
#   ../encoded/myshot.bin  ->  ../decoded/myshot.png   (sidecar ../encoded/myshot.json auto-found)
```

Options:

```bash
python decode.py myshot.bin -o out.png             # explicit output path
python decode.py /any/path/photo.bin               # full path works too
python decode.py myshot.bin --sidecar meta.json    # explicit sidecar
python decode.py myshot.bin --device cpu           # no GPU (slow: minutes per 2K image)
```

`--device` is `auto` by default (GPU if present, else CPU). Force it with `--device cuda`
or `--device cpu`.

`-o` accepts a full path or a bare name (→ `../decoded/<name>.png`); `--decoded-dir` changes the
destination. Keep each `.bin` next to its `.json`. If the sidecar is missing it still decodes —
just the full padded canvas with no crop.

---

## 4. Smoke test (no flags, uses the bundled weights)

```bash
python decode.py sample/0801.bin     -o /tmp/0801.png   # DIV2K example (legacy sidecar)
python decode.py sample/s23_0801.bin -o /tmp/s23.png    # phone-shape 2048x1536 + phone sidecar
```

Each should print a `decoded ...` summary, exit 0, and write a ~2K PNG. If these work, the
decoder is ready for real `.bin` files from the encoder.

---

## 5. Folder contents

```
decode.py            entry point (give it the .bin)
requirements.txt     Python deps (torch 2.1.2+cu121, diffusers 0.25.1, ...)
aeic/src/...         the AEIC source the decoder imports + MLCodec_{rans,CXX}.so
cpp_src/             C++ source to rebuild the .so (only if your arch/python differs, §6)
sample/              two ready-to-decode .bin/.json pairs
weights/             bundled model weights (~4.4 GB)
setup.sh             optional: re-fetch weights if deleted
```

It decodes any AEIC `.bin` — whether produced by `run_onnx_npy.py` on the desktop or by the
Android encoder app on the phone. The latent grid is taken from the `.bin` header, so the
same decoder handles the 2048×2048, 2048×1536 and (encoder `-hd`) 1024×768 shapes with no
flag or model change on this side.

---

## 6. Rebuilding the rANS `.so` (only if not x86_64 + Python 3.10)

The entropy coder ships as prebuilt `aeic/src/codec/MLCodec_{rans,CXX}.cpython-310-x86_64-linux-gnu.so`.
On a different arch/Python you'll get an `ImportError`; rebuild from the bundled source:

```bash
cmake -S cpp_src -B cpp_build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp_build -j
# copy the produced MLCodec_rans*.so and MLCodec_CXX*.so into aeic/src/codec/
```

If the build fails on warnings, drop `-Werror` from `cpp_src/CMakeLists.txt`.

---

## 7. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `torch reports no usable CUDA GPU` **but `nvidia-smi` shows one** | Wrong env, or a CPU-only / mismatched torch wheel. Activate the env from the top-level `environment-minimal.yml` and make sure torch is the cu121 build. A torch built for a CUDA newer than your driver prints "NVIDIA driver is too old" and falls back to no-GPU. |
| `NVIDIA driver is too old` warning | Same as above — you're in an env whose torch needs a newer CUDA than your driver. The `aeic` env (torch 2.1.2+cu121) matches CUDA 12.x drivers. |
| CUDA out of memory | A 2K decode peaks ~20 GB; pick a free GPU with `CUDA_VISIBLE_DEVICES=<n>`, or decode a smaller image. |
| `ImportError: MLCodec_rans...` / wrong Python | The `.so` is CPython 3.10 — use the `aeic` env (py3.10), or rebuild the `.so` (§6). |
| diffusers `ImportError` (unet_2d_blocks, transformer_2d) | `pip install diffusers==0.25.1` exactly. |
| CUDA out of memory | Decode a smaller image, or free VRAM. |
| `missing ... weights/` | A weight file was deleted — restore it or run `setup.sh`. |
