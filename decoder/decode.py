#!/usr/bin/env python
"""Home-side AEIC-SE ft2 decoder: turn a phone-produced .bin (+ .json sidecar) back
into a PNG via the full pipeline (rANS entropy-decode + 1-step diffusion + tiled VAE),
then crop / rotate / upscale per the sidecar.

This is the portable, self-contained version of the project's AEIC/e2e_decode.py. The
neural decode logic (build_net + net.decompress) is unchanged; only the file paths and
the command line were generalized so it runs from anywhere on a GPU box.

Requirements (see README.md): the `aeic` conda env (Python 3.10, torch 2.1.2+cu121,
diffusers 0.25.1) and the bundled weights. Runs on an NVIDIA CUDA GPU by default; pass
`--device cpu` to run without a GPU (works, but SLOW — several minutes for a 2K image).

Inside the repo layout the .bin is read from ../encoded/ and the PNG is
written to ../decoded/ by default.

Usage:
    python decode.py myshot.bin                # ../encoded/myshot.bin -> ../decoded/myshot.png
    python decode.py myshot.bin -o out.png     # explicit output path
    python decode.py sample/0801.bin -o /tmp/0801.png      # bundled smoke-test sample
    python decode.py myshot.bin --sidecar meta.json
    # weights default to ./weights/ (bundled); override per file: --codec-ckpt /path/AEIC_SE_ft2.pkl
"""
import os
import sys
import math
import json
import types
import argparse

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
AEIC_ROOT = os.path.join(HERE, "aeic")          # contains src/  -> `from src.codec...`
AEIC_SRC = os.path.join(AEIC_ROOT, "src")        # modules root  -> `import AEIC_practical`
# Both roots are required: codec_practical.py uses absolute `from src.codec...` imports
# while AEIC_practical.py uses bare `import base_model` / `from codec...`. This mirrors
# the original PYTHONPATH=AEIC:AEIC/src pair.
for p in (AEIC_SRC, AEIC_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# Default weight locations (overridable via CLI flag or env var, see resolve()).
DEFAULT_SD = os.path.join(HERE, "weights", "sd-turbo")
DEFAULT_CODEC = os.path.join(HERE, "weights", "AEIC_SE_ft2.pkl")
DEFAULT_VAE = os.path.join(HERE, "weights", "halfDecoder.ckpt")

# Repo folders: encoded .bin in -> decoded PNG out.
WORKSPACE = os.path.normpath(os.path.join(HERE, ".."))
ENCODED_DIR = os.path.join(WORKSPACE, "encoded")
DECODED_DIR = os.path.join(WORKSPACE, "decoded")


def resolve(cli_value, env_name, default):
    """CLI flag > environment variable > bundled default."""
    return cli_value or os.environ.get(env_name) or default


def build_net(sd_path, codec_ckpt, vae_decoder, device="cuda"):
    """Build the AEIC decoder net on `device` ("cuda" or "cpu"). Same as
    AEIC/e2e_decode.py:build_net() but the weight paths and device are parameters.
    `device` is read by base_model to place the UNet/VAE (no hardcoded .cuda()), and
    xformers (CUDA-only) is disabled on CPU."""
    from AEIC_practical import AEIC  # imported here so path setup above is in effect
    import my_utils.devices as devices   # the tiled-VAE helper reads devices.device
    devices.device = torch.device(device)   # pin it (works even if a GPU is also present)
    a = types.SimpleNamespace(
        sd_path=sd_path, codec_path=codec_ckpt, vae_decoder_path=vae_decoder,
        codec_type="AEIC-SE", device=device,
        lora_rank_unet=32, enable_xformers_memory_efficient_attention=(device == "cuda"),
        color_fix=True, use_practical_entropy_coding=True, merge_LoRA=True,
        compile_model="", use_tiled_vae=True, use_tiled_unet=True,
        vae_decoder_tiled_size=160, latent_tiled_size=192, latent_tiled_overlap=64)
    net = AEIC(sd_path=sd_path, args=a)
    net.to(device)
    net.codec.update(force=True)
    return net


def load_sidecar(path):
    """Return (content_h, content_w, orientation, orig_wh) from a sidecar, or None.

    Supports both formats handled by the stock decoder:
      - phone:  {orientation, origW, origH, contentW, contentH}
      - legacy: {orig_h, orig_w, canvas}
    """
    if not path or not os.path.isfile(path):
        return None
    meta = json.load(open(path))
    if "contentH" in meta:
        return (meta["contentH"], meta["contentW"],
                meta.get("orientation", 0), (meta.get("origW"), meta.get("origH")))
    if "orig_h" in meta:
        return (meta["orig_h"], meta["orig_w"], 0, None)
    return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bin", help="the .bin to decode (path, or a bare name looked up in ../encoded/)")
    ap.add_argument("-o", "--output", default=None,
                    help="output PNG (a path, or a bare name written to ../decoded/<name>.png; "
                         "default: ../decoded/<bin>.png)")
    ap.add_argument("--decoded-dir", default=DECODED_DIR,
                    help="where to write the PNG (default: ../decoded)")
    ap.add_argument("--sidecar", default=None,
                    help="path to the .json sidecar (default: <bin>.json next to the .bin)")
    ap.add_argument("--sd-path", default=None, help="sd-turbo dir (env AEIC_SD_PATH)")
    ap.add_argument("--codec-ckpt", default=None,
                    help="AEIC_SE_ft2.pkl path (env AEIC_CODEC_CKPT)")
    ap.add_argument("--vae-decoder", default=None,
                    help="halfDecoder.ckpt path (env AEIC_VAE_DECODER)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                    help="auto (default): GPU if available, else CPU. cpu: force CPU "
                         "(no GPU needed, but SLOW — minutes per 2K image).")
    args = ap.parse_args()

    # --- Environment check: the bundled rANS .so is built for CPython 3.10. ---
    if sys.version_info[:2] != (3, 10):
        print(f"WARNING: running under Python {sys.version_info[0]}.{sys.version_info[1]}, "
              "but the bundled rANS .so requires CPython 3.10. You are almost certainly in the "
              "wrong environment — use the `aeic` env (python 3.10, torch 2.1.2+cu121):\n"
              "  conda activate aeic   # then: python decode.py ...\n"
              "  or point directly at that env's python: <env>/bin/python decode.py ...",
              file=sys.stderr)

    # --- Resolve device. The diffusion UNet + VAE run here; default to GPU. ---
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("ERROR: --device cuda but torch reports no usable CUDA GPU "
              "(torch.cuda.is_available() == False).\n"
              "If `nvidia-smi` DOES show a GPU, this is almost always a wrong-environment / "
              "driver-mismatch problem, not a missing GPU:\n"
              "  - Use the `aeic` env (python 3.10, torch 2.1.2+cu121). A torch built for a CUDA "
              "newer than your driver prints 'NVIDIA driver is too old' and falls back to no-GPU.\n"
              "    conda activate aeic-min   (see the top-level environment-minimal.yml)\n"
              "  - To run without a GPU, pass --device cpu (much slower).\n"
              "See README.md.", file=sys.stderr)
        return 2
    if device == "cpu":
        print("NOTE: running on CPU (--device cpu). The diffusion UNet + tiled VAE are heavy; "
              "expect this to take SEVERAL MINUTES for a 2K image and a lot of RAM.",
              file=sys.stderr)

    # resolve the .bin: explicit path, else a bare name looked up in ../encoded/
    bin_path = args.bin if os.path.isfile(args.bin) else os.path.join(ENCODED_DIR, args.bin)
    bin_path = os.path.abspath(bin_path)
    if not os.path.isfile(bin_path):
        print(f"ERROR: .bin not found: {args.bin} (also tried {ENCODED_DIR}/)", file=sys.stderr)
        return 2
    stem = os.path.splitext(bin_path)[0]
    sidecar = args.sidecar or (stem + ".json")

    # output: a given path is used as-is; a bare name goes to ../decoded/<name>.png;
    # default is ../decoded/<bin-stem>.png.
    if args.output and (os.sep in args.output or args.output.lower().endswith(".png")):
        out_png = args.output
    else:
        name = args.output or os.path.splitext(os.path.basename(bin_path))[0]
        out_png = os.path.join(args.decoded_dir, name + ".png")
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)

    sd_path = resolve(args.sd_path, "AEIC_SD_PATH", DEFAULT_SD)
    codec_ckpt = resolve(args.codec_ckpt, "AEIC_CODEC_CKPT", DEFAULT_CODEC)
    vae_decoder = resolve(args.vae_decoder, "AEIC_VAE_DECODER", DEFAULT_VAE)
    for label, p in [("sd-turbo", sd_path), ("codec ckpt", codec_ckpt),
                     ("vae decoder", vae_decoder)]:
        if not os.path.exists(p):
            print(f"ERROR: missing {label}: {p}\nRun ./setup.sh to fetch the weights, "
                  f"or point to them with the matching flag/env var.", file=sys.stderr)
            return 2

    from my_utils.compress_utils import my_read_body

    net = build_net(sd_path, codec_ckpt, vae_decoder, device)
    with open(bin_path, "rb") as f:
        strings, shape = my_read_body(f)          # shape = padded canvas from .bin header
    # Grid is derived from the .bin header, so one decoder handles any encoder shape
    # (2048x2048, S23 2048x1536, ...). Unchanged from the stock decoder.
    py_h = math.ceil(shape[0] / 64) * 2
    py_w = math.ceil(shape[1] / 64) * 2
    z_size = (1, net.codec.y_channel // 2, math.ceil(py_h / 4), math.ceil(py_w / 4))

    meta = load_sidecar(sidecar)
    if meta is None:
        # No sidecar: decode the full padded canvas, no crop/rotate/upscale.
        oh, ow, orientation, orig = shape[0], shape[1], 0, None
        print(f"note: no sidecar at {sidecar} — decoding full {ow}x{oh} canvas (no crop)")
    else:
        oh, ow, orientation, orig = meta

    net.codec.entropy_coder.reset()
    net.codec.entropy_coder.set_stream(strings)
    with torch.no_grad():
        out_img, _ = net.decompress(z_size, py_h, py_w)   # entropy decode + diffusion + VAE
    out_img = (out_img.clamp(-1, 1) * 0.5 + 0.5)[0, :, :oh, :ow].float().cpu()
    rec = Image.fromarray((out_img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
    if orientation == 90:      # undo the portrait->landscape rotation applied at encode
        rec = rec.rotate(-90, expand=True)
    if orig and orig[0] and orig[1]:   # optional: upscale back to the original resolution
        rec = rec.resize((orig[0], orig[1]), Image.LANCZOS)
    rec.save(out_png)

    size = os.path.getsize(bin_path)
    bpp = size * 8 / (ow * oh)
    print(f"decoded {os.path.basename(bin_path)}: content {ow}x{oh} orient {orientation}  "
          f".bin {size} B ({bpp:.4f} bpp)  ->  {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
