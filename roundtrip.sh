#!/usr/bin/env bash
# One image, end to end: encode it to a .bin, then decode that .bin back to a PNG.
#
#   ./roundtrip.sh my_photo.jpg            # 2048x1536 canvas (default)
#   ./roundtrip.sh my_photo.jpg -fhd       # 1440x1080
#   ./roundtrip.sh my_photo.jpg -hd        # 1024x768 (smaller + much faster)
#   ./roundtrip.sh my_photo.jpg -sd        # 640x480  (smallest + fastest)
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
SIZE=""
case "${1:-}" in
  -sd|-hd|-fhd) SIZE="$1" ;;
  "") ;;
  *) echo "unknown size flag: $1 (use -sd, -hd or -fhd)" >&2; exit 1 ;;
esac

PY="${PYTHON:-python}"
NAME="$(basename "${IMAGE%.*}")${SIZE:+_${SIZE#-}}"

echo "=== 1/2  encode  (CPU) ======================================================"
( cd encoder && "$PY" run_onnx_npy.py $SIZE -i "$IMAGE" -o "$NAME" --provider cpu )

echo
echo "=== 2/2  decode  (GPU) ======================================================"
( cd decoder && "$PY" decode.py "$NAME.bin" )

echo
echo "done:"
echo "  bitstream : encoded/$NAME.bin  ($(stat -c%s "encoded/$NAME.bin") bytes)"
echo "  decoded   : decoded/$NAME.png"
echo "  original  : $IMAGE"
