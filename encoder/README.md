# AEIC-SE ft2 encoder (phone side)

The **encoder** half of this repo. `run_onnx_npy.py` does the whole
encode: it runs the neural model (also a RAM / compatibility check) **and** the rANS entropy
coding, writing the **real compressed `.bin`** (+ a `.json` sidecar) into `../encoded/`.

```bash
python run_onnx_npy.py -i input_2048x1536.png -o myshot --provider cpu
#   ../original/input_2048x1536.png  ->  ../encoded/myshot.bin (+ myshot.json)

python run_onnx_npy.py -hd -i input_2048x1536.png -o myshot_hd --provider cpu
#   same, but on the 1024x768 canvas (quarter the pixels)

python run_onnx_npy.py -fhd -i input_2048x1536.png -o myshot_fhd --provider cpu   # 1440x1080
python run_onnx_npy.py -sd  -i input_2048x1536.png -o myshot_sd  --provider cpu   # 640x480
```

- `-i/--input`: a path, or a bare filename looked up in `../original/`. Accepts a preprocessed
  `.npy` ([1,3,1536,2048]) or an ordinary image (`.png/.jpg/...`) — images are EXIF-uprighted,
  rotated portrait→landscape if needed, and resized to the model's 2048×1536 canvas, with the
  original size/orientation saved in the sidecar so the decoder restores it.
- `-o/--output`: the result name → `../encoded/<name>.bin` (+ `.json`). Omit it to reuse the
  input's name. `--encoded-dir` changes the destination.
- `-fhd` / `-hd` / `-sd`: encode at **1440×1080** / **1024×768** / **640×480** (each
  has its own model file) instead of 2048×1536. Use at most one. See [Canvases](#canvases) below.
- `--provider {auto,cpu,gpu}`: onnxruntime execution provider (default `auto`).
- `--model PATH`: use a specific `.onnx`; its input shape overrides the size flag.
- `--cdf PATH`: entropy tables (default `aeic_ft2_cdf.bin` in this folder).
- `--no-encode`: run only the model (RAM test), no `.bin`.

## Canvases

The ONNX encoders are **fixed-shape**, so each canvas is a separate model file exported from the
same `AEIC_SE_ft2.pkl` checkpoint. The flag picks the model and the resize target:

| | default (2K) | `-fhd` | `-hd` | `-sd` |
|---|---|---|---|---|
| Content (W×H) | 2048×1536 | 1440×1080 | 1024×768 | 640×480 |
| Model canvas | 2048×1536 | 1472×1088 | 1024×768 | 640×512 |
| Model | `…_2048x1536.onnx` (87 MB) | `…_1472x1088.onnx` (76 MB) | `…_1024x768.onnx` (70 MB) | `…_640x512.onnx` (67 MB) |
| Latents | `z [1,128,12,16]`, `y [1,256,48,64]` | `z [1,128,9,12]`, `y [1,256,34,46]` | `z [1,128,6,8]`, `y [1,256,24,32]` | `z [1,128,4,5]`, `y [1,256,16,20]` |
| `.bin` for the bundled test image | 10 840 B | 7 104 B | 4 306 B | 2 261 B |
| PSNR vs the 2K source | 30.10 dB | 28.94 dB | 26.65 dB | 23.61 dB |

(Model files are `aeic_se_ft2_encoder_<W>x<H>.onnx`. PSNR is measured after the decoder
upscales back to 2048×1536.)

Model canvases have to be **multiples of 64**. The decoder derives the latent grid as
`ceil(H/64)*2 × ceil(W/64)*2`, and any other size desyncs the rANS stream. 640×480 and
1440×1080 aren't multiples of 64, so the encoder resizes the image to the content size,
edge-replicates it right/bottom up to the model canvas, and stores `contentW/contentH` in the
sidecar. The decoder then crops the padding off.

Everything else is shared and unchanged: the same `aeic_ft2_cdf.bin` entropy tables, the same
pure-Python rANS step, and the **same decoder with no flag** — `decode.py` reads the canvas from
the `.bin` header (`[uint32 H][uint32 W][uint32 len]`) and derives the latent grid from it, then
upscales back to the original resolution recorded in the sidecar. A `.bin` from any canvas is
therefore a normal AEIC bitstream; nothing downstream needs to know which canvas produced it.

All four ONNX files are committed here, so nothing needs exporting to use the codec. Adding
another canvas would mean re-running the project's ONNX export script (`export_onnx_full.py H W`,
sides multiples of 64, in the upstream AEIC tree, which needs the `AEIC_SE_ft2.pkl` training
checkpoint) — that script and checkpoint are **not** part of this repo. Only ONNX exports exist
for `-fhd`/`-hd`/`-sd`; the Android app's TFLite model is still 2048x1536 only.

The rANS step is **pure Python (numpy only)** — it reads `aeic_ft2_cdf.bin` and is
**byte-identical** to the project's reference encoder and the Android app's output (verified).
No torch, no model checkpoint.

## Where the decoder fits

This codec is split across two machines. The encoder (this folder) runs on the **phone**; the
**decoder** is the sibling **`../decoder/`** folder, run on a home PC with an **NVIDIA GPU**
(self-contained, weights bundled). Round trip:
**`original/` → encode → `encoded/*.bin` → decode → `decoded/*.png`.** See the top-level `README.md`.

## Contents

| File | What |
|------|------|
| `aeic_se_ft2_encoder_2048x1536.onnx` (87 MB) | The encoder, ONNX — default canvas. |
| `aeic_se_ft2_encoder_1472x1088.onnx` (76 MB) | Same encoder exported for `-fhd` (1440×1080 content). |
| `aeic_se_ft2_encoder_1024x768.onnx` (70 MB) | Same encoder exported for the `-hd` canvas. |
| `aeic_se_ft2_encoder_640x512.onnx` (67 MB) | Same encoder exported for `-sd` (640×480 content). |
| `aeic_se_ft2_encoder_2048x1536.tflite` (87 MB) | Same encoder, TFLite/LiteRT (also serves the GPU delegate). |
| `aeic_ft2_cdf.bin` (0.8 MB) | Entropy tables — needed for the rANS step (and by the Android app). |
| `run_onnx_npy.py` | Runs the model + rANS → prints the 9 tensors and writes the `.bin`. |
| `aeic_entropy.py` | The pure-Python rANS encoder `run_onnx_npy.py` calls (numpy only). |

Input images live in `../original/` (e.g. the provided `input_2048x1536.png`/`.npy`). Both
models are **fixed-shape** and take a **single input** — no `group_map`, nothing else to build.
`-fhd`/`-hd`/`-sd` just swap in a different-size export of the same weights.

## Which scenario are you running? (IMPORTANT — pick one)

**Scenario 1 — your own ONNX/TFLite script ("just needs a model file"):**
- Feed `../original/input_2048x1536.npy` directly (or load `../original/input_2048x1536.png` and `/255`). See below.
- If you only run the **model**, you don't need `aeic_ft2_cdf.bin`. To also produce the real
  `.bin`, run `run_onnx_npy.py` (it does the rANS step and reads `aeic_ft2_cdf.bin`).

**Scenario 2 — a phone app:** the Android encoder app is **not part of this repo**. It needs
the same three asset files you see here (`*.onnx`, the matching `*.tflite`, and
`aeic_ft2_cdf.bin`) in `app/src/main/assets/`, and it preprocesses raw images itself. Only the
2048x1536 TFLite export exists; there is no `-fhd`/`-hd`/`-sd` TFLite yet.

## Input contract (what the model expects)

- **Shape:** `[1, 3, 1536, 2048]`  (NCHW: batch 1, RGB, H=1536, W=2048). With `-fhd`:
  `[1, 3, 1088, 1472]`, `-hd`: `[1, 3, 768, 1024]`, `-sd`: `[1, 3, 512, 640]`
- **dtype:** `float32`
- **Value range:** `[0.0, 1.0]`  (i.e. pixel/255)
- **Normalization is inside the model** — do NOT subtract mean / divide by std yourself.
  Just pass pixels in `[0,1]`.
- ONNX input name: `image_rgb_01`
- TFLite input name: `serving_default_args_0`

The image is a real 2K photo (DIV2K 0803, 2040×1536) replicate-padded on the right edge to
2048 wide. The padding is already baked into the provided files.

## Two ways to feed it (use whichever your script already supports)

**A. Load the ready tensor (`.npy`):**
```python
import numpy as np
x = np.load("input_2048x1536.npy")   # float32, [1,3,1536,2048], already in [0,1]
# ONNX:
out = session.run(None, {"image_rgb_01": x})
```

**B. Load the PNG and normalize (if your script takes an image path):**
```python
import numpy as np
from PIL import Image
img = np.array(Image.open("input_2048x1536.png").convert("RGB"), np.float32) / 255.0
x = img.transpose(2, 0, 1)[None]      # HWC -> [1,3,1536,2048]
```
Both routes produce a **bit-identical** tensor (verified, max diff = 0.0).

## Output

**Model:** 9 tensors (the quantized latents + entropy params): one `z` `[1,128,12,16]` and
eight `[1,256,48,64]` (other canvases: see the [Canvases](#canvases) table). A successful
run that returns these shapes = the model ran fine.

**Compressed file (`run_onnx_npy.py`):** after the model, the script entropy-codes those 9
tensors into `../encoded/`:
- `../encoded/<name>.bin` — the real compressed bitstream (`<name>` from `-o`, else the input
  name). Header `[uint32 H][uint32 W][uint32 len]` (big-endian, padded canvas dims) + the rANS
  stream. ~11 KB here (≈0.028 bpp, ~300× smaller than the source 2K image).
- `../encoded/<name>.json` — sidecar with the original size + orientation so the decoder
  crops/rotates/upscales back. For an image input it's filled automatically; for a `.npy` it
  records the full 2048×1536 canvas.

Decode it on a GPU box with `cd ../decoder && python decode.py <name>.bin` → `../decoded/<name>.png`
(see `../decoder/README.md`). The decoder is shape- and source-agnostic — it also accepts the
Android app's `.bin`.

## Expected memory (measured on desktop x86, for reference)

At this 2048×1536 shape, peak working set during inference:

| Runtime | Peak RAM |
|---------|----------|
| ONNX (CPU) | ~1.6–3.0 GB |
| TFLite (CPU) | ~1.9 GB |

The model **weights are only ~90 MB** — the multi-GB peak is the intermediate activation
feature maps, which scale with image resolution. So on a phone:
- A device with **< ~3 GB free RAM may OOM/crash** with the ONNX-CPU path. TFLite-CPU is a
  bit lighter. A GPU-delegate path (fp16) would be lighter still (~1 GB), if your script
  supports it.
- If it crashes, the first thing to try is a **smaller input** (less RAM), then a lighter
  runtime/delegate.
