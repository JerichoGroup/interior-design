#!/usr/bin/env python3
"""Pure-Python rANS entropy coder for the AEIC-SE ft2 encoder — the step that turns
the model's 9 latent tensors into the real compressed `.bin`.

This is the "everything after the neural model" half of the encoder. It depends on
**numpy only** (no torch, no model checkpoint): it reads the entropy tables from
`aeic_ft2_cdf.bin` and reproduces, in Python, exactly what the C++ rANS coder and the
verified Kotlin host (`AeicHost.kt` / `CdfTables.kt` / `Rans.kt`) do:

  squeeze 256->64 channels  ->  build CDF indexes from `scales`  ->  rANS encode
  (z then y0..3, 2 substreams)  ->  prepend the big-endian .bin header

The output is byte-identical to the project's reference encoder (AEIC/e2e_encode.py)
and to the Android app's `.bin`. The rANS classes below are a verbatim copy of
AEIC/rans_reference.py (which is asserted byte-identical to the C++ `.so`).
"""
import struct
import numpy as np

# --- rANS constants (rans.cpp) ---------------------------------------------------
PRECISION = 16
RANS_L = 1 << 23
BYPASS_PRECISION = 2
MAX_BYPASS_VAL = (1 << BYPASS_PRECISION) - 1


# =============================================================================
# rANS encoder — verbatim from AEIC/rans_reference.py (pure python, numpy only).
# =============================================================================
class RansEncoderLib:
    """Mirror of RansEncoderLib (rans.cpp): accumulates _syms across calls, flush -> bytes."""
    def __init__(self):
        self.rans_symbols = []
        self.sizes = []
        self.offsets = []
        self._syms = []

    def add_cdf(self, cdfs, cdf_sizes, offsets):
        ransSym = []
        for row in cdfs:
            r = [(int(row[j]), int(row[j + 1]) - int(row[j])) for j in range(len(row) - 1)]
            ransSym.append(r)
        self.rans_symbols.append(ransSym)
        self.sizes.append(np.asarray(cdf_sizes, dtype=np.int64))
        self.offsets.append(np.asarray(offsets, dtype=np.int64))
        return len(self.rans_symbols) - 1

    def reset(self):
        self._syms = []

    def encode_with_indexes(self, symbols, indexes, group):
        symbols = np.asarray(symbols, dtype=np.int64)
        indexes = np.asarray(indexes, dtype=np.int64)
        sizes = self.sizes[group]; offsets = self.offsets[group]
        ransSym = self.rans_symbols[group]
        syms = self._syms
        for i in range(symbols.shape[0]):
            cdf_idx = int(indexes[i])
            if cdf_idx < 0:
                continue
            max_value = int(sizes[cdf_idx]) - 2
            value = int(symbols[i]) - int(offsets[cdf_idx])
            raw_val = 0
            if value < 0:
                raw_val = -2 * value - 1
                value = max_value
            elif value >= max_value:
                raw_val = 2 * (value - max_value)
                value = max_value
            syms.append(ransSym[cdf_idx][value])
            if value == max_value:
                n_bypass = 0
                while (raw_val >> (n_bypass * BYPASS_PRECISION)) != 0:
                    n_bypass += 1
                val = n_bypass
                while val >= MAX_BYPASS_VAL:
                    syms.append((MAX_BYPASS_VAL, 0))
                    val -= MAX_BYPASS_VAL
                syms.append((val, 0))
                for j in range(n_bypass):
                    val1 = (raw_val >> (j * BYPASS_PRECISION)) & MAX_BYPASS_VAL
                    syms.append((val1, 0))

    def flush(self):
        x = RANS_L
        buf = bytearray(len(self._syms) * 4 + 1024)
        ptr = len(buf)
        for (start, rng) in reversed(self._syms):
            if rng != 0:
                x_max = rng << 15
                while x >= x_max:
                    ptr -= 1; buf[ptr] = x & 0xFF; x >>= 8
                x = ((x // rng) << PRECISION) + (x % rng) + start
            else:
                freq = 1 << (PRECISION - BYPASS_PRECISION)
                x_max = freq << 15
                while x >= x_max:
                    ptr -= 1; buf[ptr] = x & 0xFF; x >>= 8
                x = (x << BYPASS_PRECISION) | start
        ptr -= 4
        buf[ptr] = x & 0xFF; buf[ptr + 1] = (x >> 8) & 0xFF
        buf[ptr + 2] = (x >> 16) & 0xFF; buf[ptr + 3] = (x >> 24) & 0xFF
        return bytes(buf[ptr:])


class RansEncoder:
    """Mirror of py_rans.cpp RansEncoder with streamPart substreams (default 2)."""
    def __init__(self, stream_part=2):
        self.encoders = [RansEncoderLib() for _ in range(stream_part)]

    def add_cdf(self, cdfs, cdf_sizes, offsets):
        idx = 0
        for e in self.encoders:
            idx = e.add_cdf(cdfs, cdf_sizes, offsets)
        return idx

    def reset(self):
        for e in self.encoders:
            e.reset()

    def encode_with_indexes(self, symbols, indexes, group):
        symbols = np.asarray(symbols); indexes = np.asarray(indexes)
        n = symbols.shape[0]; enc_n = len(self.encoders)
        each = n // enc_n
        for i, e in enumerate(self.encoders):
            size = each if i < enc_n - 1 else n - each * (enc_n - 1)
            off = i * each
            e.encode_with_indexes(symbols[off:off + size], indexes[off:off + size], group)

    def encode_with_z4y_indexes(self, z_sym, z_idx, z_group, y_syms, y_idx, y_group):
        self.encode_with_indexes(z_sym, z_idx, z_group)
        for s, ix in zip(y_syms, y_idx):
            self.encode_with_indexes(s, ix, y_group)

    def flush_and_get_stream(self):
        results = [e.flush() for e in self.encoders]
        enc_n = len(results)
        maximum = max((len(results[i]) for i in range(enc_n - 1)), default=0)
        per_hdr = 2 if maximum <= 65535 else 4
        out = bytearray()
        flag = ((enc_n - 1) << 4) + (1 if per_hdr == 2 else 0)
        out.append(flag)
        for i in range(enc_n - 1):
            out += int(len(results[i])).to_bytes(per_hdr, "little")
        for i in range(enc_n):
            out += results[i]
        return bytes(out)


# =============================================================================
# Host pipeline — ports of CdfTables.load / AeicHost.{sequeeze,buildYIndex,encode}.
# =============================================================================
class CdfTables:
    """Parsed aeic_ft2_cdf.bin (see AEIC/export_cdf_tables.py for the layout)."""
    def __init__(self, precision, y_channel, scales_levels, log_scale_min,
                 log_scale_step, bypass_threshold, scale_floor, z_group, y_group):
        self.precision = precision
        self.y_channel = y_channel
        self.scales_levels = scales_levels
        self.log_scale_min = log_scale_min
        self.log_scale_step = log_scale_step
        self.bypass_threshold = bypass_threshold
        self.scale_floor = scale_floor
        self.z_group = z_group   # (cdf[n,maxlen], cdf_length[n], offset[n])
        self.y_group = y_group

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            b = f.read()
        if b[:8] != b"AEICDF01":
            raise ValueError(f"{path}: bad magic (not an AEIC cdf file)")
        off = 8
        precision, y_channel, scales_levels = struct.unpack_from("<iii", b, off); off += 12
        log_min, log_step, bypass_thr, floor = struct.unpack_from("<dddd", b, off); off += 32

        def read_group(off):
            n, maxlen = struct.unpack_from("<ii", b, off); off += 8
            cdf = np.frombuffer(b, "<i4", n * maxlen, off).reshape(n, maxlen).copy(); off += 4 * n * maxlen
            length = np.frombuffer(b, "<i4", n, off).copy(); off += 4 * n
            offset = np.frombuffer(b, "<i4", n, off).copy(); off += 4 * n
            return (cdf, length, offset), off

        z_group, off = read_group(off)
        y_group, off = read_group(off)
        return CdfTables(precision, y_channel, scales_levels, log_min, log_step,
                         bypass_thr, floor, z_group, y_group)

    def build_y_index(self, scales):
        """my_build_indexes (codec_practical.py:212): squeezed scale -> int16 cdf idx (or -1)."""
        s = np.maximum(scales.astype(np.float64), self.scale_floor)
        idx = (np.log(s) - self.log_scale_min) / self.log_scale_step
        idx = np.clip(idx, 0.0, self.scales_levels - 1)
        out = idx.astype(np.int32)                                   # truncate toward zero
        out = np.where(scales < self.bypass_threshold, -1, out)
        return out.astype(np.int16)


def _sequeeze(arr_flat, channels, hw):
    """sum the 4 contiguous channel-chunks (256 -> 64). Matches codec.sequeeze:
    (g1+g2)+(g3+g4); numpy reduces the length-4 axis sequentially -> same result."""
    cq = channels // 4
    return arr_flat.reshape(4, cq * hw).sum(axis=0)


def encode_to_bin(outputs_by_name, cdf, pad_h, pad_w):
    """Turn the encoder's 9 named outputs into a complete `.bin` (header + rANS stream).

    outputs_by_name: dict with keys z_q, y_q_0..3, scales_0..3 (numpy float arrays,
                     each [1,C,H,W] as returned by the ONNX/TFLite model).
    cdf:             CdfTables.load(aeic_ft2_cdf.bin)
    pad_h, pad_w:    the padded canvas dims (the model's input H, W) for the header.
    Returns: (bin_bytes, stream_len).
    """
    z = np.asarray(outputs_by_name["z_q"]).reshape(-1)
    z_ch = np.asarray(outputs_by_name["z_q"]).shape[1]
    z_hw = z.size // z_ch
    y_shape = np.asarray(outputs_by_name["y_q_0"]).shape       # [1, C, H, W]
    y_ch, y_hw = y_shape[1], y_shape[2] * y_shape[3]

    z_sym = z.astype(np.int16)
    z_idx = np.repeat(np.arange(z_ch, dtype=np.int16), z_hw)   # get_compress_info: index == channel

    y_syms, y_idx = [], []
    for i in range(4):
        yq = np.asarray(outputs_by_name[f"y_q_{i}"]).reshape(-1)
        sc = np.asarray(outputs_by_name[f"scales_{i}"]).reshape(-1)
        y_syms.append(_sequeeze(yq, y_ch, y_hw).astype(np.int16))
        y_idx.append(cdf.build_y_index(_sequeeze(sc, y_ch, y_hw)))

    enc = RansEncoder(2)
    zg = enc.add_cdf(*cdf.z_group)        # -> 0 (z_cdf_group_index)
    yg = enc.add_cdf(*cdf.y_group)        # -> 1 (y_cdf_group_index)
    enc.encode_with_z4y_indexes(z_sym, z_idx, zg, y_syms, y_idx, yg)
    stream = enc.flush_and_get_stream()

    header = struct.pack(">III", int(pad_h), int(pad_w), len(stream))   # my_write_body (big-endian)
    return header + stream, len(stream)
