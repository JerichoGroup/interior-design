#!/usr/bin/env python3
"""
AEIC-SE ft2 encoder: take an image from ../original/, run the neural encoder, and write
the real compressed .bin (+ .json sidecar) into ../encoded/.

It does two things:
  1. Runs the neural encoder (also a RAM / compatibility check) and prints the 9 output
     tensors. It reads what the model actually wants (input name "image_rgb_01", float32,
     shape [1,3,H,W], C-contiguous) and resizes/coerces the input to match.
  2. Runs the rANS entropy coding (pure Python, numpy only — see aeic_entropy.py) on those
     tensors to write the actual compressed .bin. Byte-identical to the project's reference
     encoder and the Android app's output. Use --no-encode to run only the model (no .bin).

Canvases (the ONNX encoders are fixed-shape, so each canvas is its own model file):
    default   2048x1536 ("2K")  -> aeic_se_ft2_encoder_2048x1536.onnx
    -fhd      1440x1080 ("FHD") -> aeic_se_ft2_encoder_1472x1088.onnx  (edge-padded canvas)
    -hd       1024x768  ("HD")  -> aeic_se_ft2_encoder_1024x768.onnx
    -sd       640x480   ("SD")  -> aeic_se_ft2_encoder_640x512.onnx    (edge-padded canvas)
Canvases must be multiples of 64 (decoder latent-grid rule), so SD/FHD content is padded
right/bottom and the sidecar tells the decoder to crop it off.
Smaller canvases mean a smaller .bin and less encoder RAM/time at lower fidelity (the
decoder upscales back to the original size via the sidecar).
The decoder needs no flag — it reads the canvas from the .bin header.

Inputs (../original/): a preprocessed .npy ([1,3,H,W], float32, [0,1], matching the chosen
canvas) OR an ordinary image (.png/.jpg/...). Images are EXIF-uprighted, rotated
portrait->landscape if needed, and resized to the model's canvas; the sidecar records the
original size/orientation so the decoder restores it.

Usage:
    python run_onnx_npy.py                              # encodes ../original/input_2048x1536.png
    python run_onnx_npy.py -i photo.jpg -o myshot       # ../original/photo.jpg -> ../encoded/myshot.bin
    python run_onnx_npy.py -hd -i photo.jpg -o myshot   # same, on the 1024x768 canvas
    python run_onnx_npy.py -sd  -i photo.jpg -o myshot  # 640x480 canvas
    python run_onnx_npy.py -fhd -i photo.jpg -o myshot  # 1440x1080 canvas
    python run_onnx_npy.py --input /abs/path.png -o out
    python run_onnx_npy.py --provider cpu               # force CPU (default: auto)
    python run_onnx_npy.py --no-encode                  # model only, no .bin
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.normpath(os.path.join(HERE, ".."))   # repo root
ORIGINAL_DIR = os.path.join(WORKSPACE, "original")
ENCODED_DIR = os.path.join(WORKSPACE, "encoded")

# The ONNX encoders are fixed-shape: one model file per canvas. (W, H) -> model.
# Model canvases must be multiples of 64: the decoder derives the latent grid as
# ceil(H/64)*2 x ceil(W/64)*2, so anything else desyncs the rANS stream.
CANVAS_2K = (2048, 1536)                                 # default
CANVAS_FHD = (1472, 1088)                                # -fhd (1440x1080 content, padded)
CANVAS_HD = (1024, 768)                                  # -hd
CANVAS_SD = (640, 512)                                   # -sd  (640x480 content, padded)
MODELS = {
    CANVAS_2K: os.path.join(HERE, "aeic_se_ft2_encoder_2048x1536.onnx"),
    CANVAS_FHD: os.path.join(HERE, "aeic_se_ft2_encoder_1472x1088.onnx"),
    CANVAS_HD: os.path.join(HERE, "aeic_se_ft2_encoder_1024x768.onnx"),
    CANVAS_SD: os.path.join(HERE, "aeic_se_ft2_encoder_640x512.onnx"),
}
# Image content size (W, H) per canvas, where it differs from the canvas: the image is
# resized to the content size and edge-padded right/bottom to the canvas; the sidecar's
# contentW/contentH make the decoder crop the padding off.
CONTENT = {CANVAS_FHD: (1440, 1080), CANVAS_SD: (640, 480)}
# flag -> canvas
FLAG_CANVAS = {"sd": CANVAS_SD, "hd": CANVAS_HD, "fhd": CANVAS_FHD}
DEFAULT_CDF = os.path.join(HERE, "aeic_ft2_cdf.bin")
DEFAULT_INPUT = os.path.join(ORIGINAL_DIR, "input_2048x1536.png")
DEFAULT_MODEL = MODELS[CANVAS_2K]        # kept for the benchmark/ scripts that import it


def resolve_input(value):
    """Accept an absolute/relative path, or a bare filename looked up in ../original/."""
    if os.path.isfile(value):
        return value
    cand = os.path.join(ORIGINAL_DIR, value)
    if os.path.isfile(cand):
        return cand
    return value  # let the caller report the not-found error


def load_input(path, canvas=CANVAS_2K, content=None):
    """Return (x [1,3,H,W] float32 in [0,1], sidecar dict) for canvas=(W, H).

    Defaults to the 2048x1536 canvas (what benchmark/ imports expect); the CLI passes the
    canvas read from the model, so `-sd/-hd/-fhd` resize to their canvas instead.

    .npy  -> used as-is (must already be the model's input shape).
    image -> EXIF-upright, rotate portrait->landscape (orientation=90), resize to `content`
             (W, H; default = canvas), then edge-pad right/bottom up to the canvas.
    """
    cw, ch = canvas
    tw, th = content or canvas
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        x = np.load(path, allow_pickle=False)
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        h, w = int(x.shape[-2]), int(x.shape[-1])
        return np.ascontiguousarray(x), {"orig_h": h, "orig_w": w, "canvas": max(h, w)}

    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    orig_w, orig_h = im.size
    orientation = 0
    if orig_h > orig_w:                      # portrait -> landscape (decoder rotates back)
        im = im.rotate(90, expand=True)
        orientation = 90
    if im.size != (tw, th):
        print(f"resizing {im.size[0]}x{im.size[1]} -> {tw}x{th} for the fixed-shape model")
        im = im.resize((tw, th), Image.LANCZOS)
    arr = np.asarray(im, np.float32) / 255.0
    if (tw, th) != (cw, ch):                 # replicate-pad up to the model canvas
        print(f"padding {tw}x{th} -> {cw}x{ch} (edge replicate; decoder crops it off)")
        arr = np.pad(arr, ((0, ch - th), (0, cw - tw), (0, 0)), mode="edge")
    x = np.ascontiguousarray(arr.transpose(2, 0, 1)[None])   # HWC -> [1,3,H,W]
    sidecar = {"orientation": orientation, "origW": orig_w, "origH": orig_h,
               "contentW": tw, "contentH": th}
    return x, sidecar


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None,
                    help="path to the .onnx encoder (default: the model for the chosen canvas)")
    size = ap.add_mutually_exclusive_group()
    size.add_argument("-sd", "--sd", dest="size", action="store_const", const="sd",
                      help="encode 640x480 content (SD; model canvas 640x512, "
                           "32 rows of edge padding the decoder crops off)")
    size.add_argument("-hd", "--hd", dest="size", action="store_const", const="hd",
                      help="encode on the 1024x768 canvas instead of 2048x1536 "
                           "(a quarter of the pixels: ~2.5x smaller .bin, ~4x less "
                           "encoder RAM/compute, lower fidelity)")
    size.add_argument("-fhd", "--fhd", dest="size", action="store_const", const="fhd",
                      help="encode 1440x1080 FullHD content (model canvas 1472x1088, "
                           "edge padding the decoder crops off)")
    ap.add_argument("-i", "--input", "--npy", dest="input", default=DEFAULT_INPUT,
                    help="input image/.npy (path, or a bare name looked up in ../original/)")
    ap.add_argument("-o", "--output", default=None,
                    help="output base name written to ../encoded/<name>.bin (+ .json); "
                         "default: the input's name")
    ap.add_argument("--encoded-dir", default=ENCODED_DIR,
                    help="where to write the .bin/.json (default: ../encoded)")
    ap.add_argument(
        "--provider", default="auto", choices=["auto", "cpu", "gpu"],
        help="execution provider: cpu=CPUExecutionProvider, gpu=CUDA/other if available, "
             "auto=let onnxruntime pick (default)")
    ap.add_argument("--no-encode", action="store_true",
                    help="stop after the model (RAM test only); do NOT produce the .bin")
    ap.add_argument("--cdf", default=DEFAULT_CDF,
                    help="entropy tables (default: aeic_ft2_cdf.bin in this folder)")
    args = ap.parse_args()

    canvas = FLAG_CANVAS.get(args.size, CANVAS_2K)
    model_path = args.model or MODELS[canvas]

    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime is not installed. Install with:\n"
              "  pip install onnxruntime        # CPU\n"
              "  pip install onnxruntime-gpu    # CUDA", file=sys.stderr)
        return 2

    print(f"onnxruntime version: {ort.__version__}")
    print(f"available providers: {ort.get_available_providers()}")

    in_path = resolve_input(args.input)
    for p, label in [(model_path, "model"), (in_path, "input")]:
        if not os.path.isfile(p):
            print(f"ERROR: {label} file not found: {p}", file=sys.stderr)
            return 2

    # --- 1. Pick provider and open the model --------------------------------
    if args.provider == "cpu":
        providers = ["CPUExecutionProvider"]
    elif args.provider == "gpu":
        avail = ort.get_available_providers()
        providers = [p for p in avail if p != "CPUExecutionProvider"] or ["CPUExecutionProvider"]
    else:  # auto
        providers = None  # onnxruntime default order

    print(f"\nmodel: {model_path}  (canvas {canvas[0]}x{canvas[1]}"
          f"{', -' + args.size if args.size else ''})")
    sess = ort.InferenceSession(model_path, providers=providers)
    print(f"using providers: {sess.get_providers()}")

    # --- 2. Read what the model ACTUALLY wants (don't hardcode/guess) --------
    inp = sess.get_inputs()[0]
    in_name = inp.name
    in_shape = inp.shape            # e.g. [1, 3, 1536, 2048]; dims may be strings if dynamic
    in_type = inp.type             # e.g. "tensor(float)"
    print(f"model input -> name={in_name!r} shape={in_shape} type={in_type}")

    # the canvas the input is resized to comes from the model itself when it is
    # fixed-shape, so --model with any other fixed-shape export works unchanged.
    if len(in_shape) == 4 and isinstance(in_shape[2], int) and isinstance(in_shape[3], int):
        model_canvas = (int(in_shape[3]), int(in_shape[2]))   # (W, H)
        if model_canvas != canvas:
            print(f"note: using the model's canvas {model_canvas[0]}x{model_canvas[1]} "
                  f"(not the flag's {canvas[0]}x{canvas[1]})")
            canvas = model_canvas

    # --- 3. Load + preprocess the input -------------------------------------
    x, sidecar = load_input(in_path, canvas, CONTENT.get(canvas))
    print(f"\nloaded input: {in_path}\n  tensor shape={x.shape} dtype={x.dtype} "
          f"min={float(x.min()):.4f} max={float(x.max()):.4f}  sidecar={sidecar}")

    # --- 4. Coerce / validate the array against the model -------------------
    want_f32 = "float" in in_type and "float16" not in in_type
    if want_f32 and x.dtype != np.float32:
        x = x.astype(np.float32)
    if len(in_shape) != x.ndim:
        print(f"ERROR: rank mismatch — model wants {len(in_shape)} dims "
              f"{in_shape}, input has {x.ndim} dims {x.shape}", file=sys.stderr)
        return 3
    for axis, (want, got) in enumerate(zip(in_shape, x.shape)):
        if isinstance(want, int) and want != got:
            print(f"ERROR: shape mismatch on axis {axis} — model wants {want}, "
                  f"input has {got} (full: model {in_shape} vs input {x.shape})",
                  file=sys.stderr)
            if os.path.splitext(in_path)[1].lower() == ".npy":
                print("hint: a .npy input is used as-is, so it must already match the "
                      "model's canvas. Pass an image instead (it is resized for you), or "
                      "pick the matching model with -sd/-hd/-fhd (or none for 2K).",
                      file=sys.stderr)
            return 3
    x = np.ascontiguousarray(x)

    # --- 5. Run ----------------------------------------------------------
    print("\nrunning inference...")
    outputs = sess.run(None, {in_name: x})

    out_meta = sess.get_outputs()
    print(f"\nOK — model returned {len(outputs)} output tensor(s):")
    for meta, arr in zip(out_meta, outputs):
        arr = np.asarray(arr)
        print(f"   {meta.name:<24} shape={tuple(arr.shape)} dtype={arr.dtype} "
              f"min={float(arr.min()):.4f} max={float(arr.max()):.4f}")

    # --- 6. Entropy-code the latents into the real compressed .bin -----------
    if args.no_encode:
        print("\n(--no-encode given: ran the model only, no .bin produced)")
        print("\nSUCCESS: the model ran.")
        return 0

    names = {m.name for m in out_meta}
    required = {"z_q", *(f"y_q_{i}" for i in range(4)), *(f"scales_{i}" for i in range(4))}
    if not required.issubset(names):
        print(f"\nWARNING: model outputs {sorted(names)} are not the AEIC encoder's "
              f"(need {sorted(required)}); skipping rANS encode.", file=sys.stderr)
        return 0
    if not os.path.isfile(args.cdf):
        print(f"\nWARNING: entropy tables not found: {args.cdf}; skipping rANS encode.",
              file=sys.stderr)
        return 0

    import aeic_entropy
    out_by_name = {m.name: np.asarray(a) for m, a in zip(out_meta, outputs)}
    pad_h, pad_w = int(x.shape[2]), int(x.shape[3])
    cdf = aeic_entropy.CdfTables.load(args.cdf)
    bin_bytes, stream_len = aeic_entropy.encode_to_bin(out_by_name, cdf, pad_h, pad_w)

    name = os.path.splitext(args.output)[0] if args.output else \
        os.path.splitext(os.path.basename(in_path))[0]
    os.makedirs(args.encoded_dir, exist_ok=True)
    out_bin = os.path.join(args.encoded_dir, name + ".bin")
    sidecar_path = os.path.join(args.encoded_dir, name + ".json")
    with open(out_bin, "wb") as f:
        f.write(bin_bytes)
    json.dump(sidecar, open(sidecar_path, "w"))

    rw = sidecar.get("origW", sidecar.get("orig_w"))
    rh = sidecar.get("origH", sidecar.get("orig_h"))
    comp = len(bin_bytes)
    bpp = comp * 8.0 / (rw * rh)
    print("\nrANS encode OK — this is the real compressed file:")
    print(f"   .bin            : {out_bin}")
    print(f"   sidecar         : {sidecar_path}")
    print(f"   compressed size : {comp} bytes ({comp / 1024:.1f} KB)  "
          f"[{stream_len} B rANS stream + 12 B header]")
    print(f"   canvas {pad_w}x{pad_h}   original {rw}x{rh}   bpp={bpp:.4f}")
    print(f"   decode it       : cd ../decoder && python decode.py {out_bin}")
    print("\nSUCCESS: the model ran and the .bin was written to ../encoded/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
