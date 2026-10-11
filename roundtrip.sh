#!/usr/bin/env bash
# One image, end to end: encode it to a .bin, then decode that .bin back to a PNG.
#
#   ./roundtrip.sh my_photo.jpg            # 2048x1536 canvas (default)
#   ./roundtrip.sh my_photo.jpg -hd        # 1024x768 canvas (smaller + much faster)
#   PYTHON=/path/to/python ./roundtrip.sh my_photo.jpg
#
# The image may be anywhere; a bare filename is looked up in ./original/.
# Encoding needs only a CPU. Decoding needs an NVIDIA GPU (use --device cpu in
# decoder/decode.py directly if you have none — it works but takes minutes).

set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

if [ $# -lt 1 ]; then
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi

IMAGE="$1"; shift
HD=""
[ "${1:-}" = "-hd" ] && HD="-hd"

PY="${PYTHON:-python}"
NAME="$(basename "${IMAGE%.*}")${HD:+_hd}"

echo "=== 1/2  encode  (CPU) ======================================================"
( cd encoder && "$PY" run_onnx_npy.py $HD -i "$IMAGE" -o "$NAME" --provider cpu )

echo
echo "=== 2/2  decode  (GPU) ======================================================"
( cd decoder && "$PY" decode.py "$NAME.bin" )

echo
echo "done:"
echo "  bitstream : encoded/$NAME.bin  ($(stat -c%s "encoded/$NAME.bin") bytes)"
echo "  decoded   : decoded/$NAME.png"
echo "  original  : $IMAGE"
