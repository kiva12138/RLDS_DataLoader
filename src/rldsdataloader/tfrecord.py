"""Pure-Python TFRecord + tf.train.Example reader.

The TFRecord format is documented at
https://www.tensorflow.org/tutorials/load_data/tfrecord; each record is
``uint64 length | uint32 crc(length) | payload | uint32 crc(payload)``.
We skip the CRC checks for speed and parse the ``tf.train.Example`` proto
directly off the wire.

TFDS stores one episode per record, with the original feature tree flattened
into keys joined by ``/`` (e.g. ``steps/observation/image``).  Per-step
tensors are concatenated along the leading axis inside a single
``FloatList`` / ``Int64List`` / ``BytesList``.
"""
from __future__ import annotations

import struct
from typing import Dict, Iterator, List, Tuple, Union


# --- protobuf wire format --------------------------------------------------

def _read_varint(buf: bytes, pos: int) -> Tuple[int, int]:
    r, s = 0, 0
    while True:
        b = buf[pos]; pos += 1
        r |= (b & 0x7F) << s
        if not (b & 0x80):
            return r, pos
        s += 7


def _iter_fields(buf: bytes) -> Iterator[Tuple[int, int, Union[int, bytes]]]:
    """Yield ``(field_number, wire_type, value)`` for every field in ``buf``."""
    pos, end = 0, len(buf)
    while pos < end:
        tag, pos = _read_varint(buf, pos)
        fn, wt = tag >> 3, tag & 7
        if wt == 0:        # varint
            v, pos = _read_varint(buf, pos)
            yield fn, wt, v
        elif wt == 2:      # length-delimited
            n, pos = _read_varint(buf, pos)
            yield fn, wt, buf[pos:pos + n]
            pos += n
        elif wt == 1:      # 64-bit fixed
            yield fn, wt, buf[pos:pos + 8]; pos += 8
        elif wt == 5:      # 32-bit fixed
            yield fn, wt, buf[pos:pos + 4]; pos += 4
        else:
            raise ValueError(f"unsupported wire type {wt}")


def _parse_feature(buf: bytes) -> Tuple[str, list]:
    """Parse a ``tf.train.Feature`` into ``(kind, values)``."""
    for fn, wt, data in _iter_fields(buf):
        if wt != 2:
            continue
        if fn == 1:                                      # BytesList
            return "bytes", [bytes(d) for f2, w2, d in _iter_fields(data)
                             if f2 == 1 and w2 == 2]
        if fn == 2:                                      # FloatList (packed)
            vals: List[float] = []
            for f2, w2, d in _iter_fields(data):
                if f2 != 1:
                    continue
                if w2 == 2:
                    vals.extend(struct.unpack(f"<{len(d)//4}f", d))
                elif w2 == 5:
                    vals.append(struct.unpack("<f", d)[0])
            return "float", vals
        if fn == 3:                                      # Int64List (packed varint)
            vals_i: List[int] = []
            for f2, w2, d in _iter_fields(data):
                if f2 != 1:
                    continue
                if w2 == 2:
                    p = 0
                    while p < len(d):
                        v, p = _read_varint(d, p)
                        if v >= (1 << 63):
                            v -= 1 << 64
                        vals_i.append(v)
                elif w2 == 0:
                    vals_i.append(d)                     # already varint-decoded
            return "int", vals_i
    return "bytes", []


def parse_example(buf: bytes) -> Dict[str, Tuple[str, list]]:
    """Parse a serialized ``tf.train.Example`` into ``{name: (kind, values)}``."""
    out: Dict[str, Tuple[str, list]] = {}
    for fn, wt, data in _iter_fields(buf):
        if fn != 1 or wt != 2:                           # Example.features
            continue
        for fn2, wt2, data2 in _iter_fields(data):       # Features.feature (map)
            if fn2 != 1 or wt2 != 2:
                continue
            key, value = None, None
            for fn3, wt3, data3 in _iter_fields(data2):  # map entry
                if fn3 == 1 and wt3 == 2:
                    key = data3.decode("utf-8")
                elif fn3 == 2 and wt3 == 2:
                    value = _parse_feature(data3)
            if key is not None and value is not None:
                out[key] = value
    return out


# --- TFRecord container ----------------------------------------------------

_HEADER = struct.Struct("<Q")


def iter_tfrecord(path: str) -> Iterator[bytes]:
    """Yield raw record payloads from a TFRecord file (CRCs are ignored)."""
    with open(path, "rb") as f:
        while True:
            head = f.read(8)
            if len(head) < 8:
                return
            length = _HEADER.unpack(head)[0]
            f.read(4)  # length CRC
            payload = f.read(length)
            f.read(4)  # payload CRC
            if len(payload) != length:
                return
            yield payload


def iter_examples(path: str) -> Iterator[Dict[str, Tuple[str, list]]]:
    """Yield parsed ``tf.train.Example`` dicts from a TFRecord file."""
    for payload in iter_tfrecord(path):
        yield parse_example(payload)
