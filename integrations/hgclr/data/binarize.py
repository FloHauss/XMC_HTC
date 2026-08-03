"""
Standalone implementation of fairseq's MMapIndexedDataset binary format.
Avoids importing fairseq directly (which has numpy compatibility issues).

Format reference: fairseq/data/indexed_dataset.py MMapIndexedDataset
"""

import struct
import numpy as np


_HDR_MAGIC = b"MMIDIDX\x00\x00"
_VERSION = 1

# dtype code 8 = uint16, used when vocab_size < 65500
_DTYPE = np.uint16
_DTYPE_CODE = 8
_DTYPE_MAX = np.iinfo(_DTYPE).max


def write_mmap_dataset(prefix, sequences):
    """
    Write sequences to fairseq mmap indexed dataset files.

    Args:
        prefix: path prefix (will produce prefix.bin and prefix.idx)
        sequences: list of lists of integers
    """
    bin_path = prefix + ".bin"
    idx_path = prefix + ".idx"

    sizes = []
    with open(bin_path, "wb") as data_file:
        for index, seq in enumerate(sequences):
            values = np.asarray(seq)
            if values.ndim != 1:
                raise ValueError(f"Sequence {index} must be one-dimensional.")
            if values.size and (values.min() < 0 or values.max() > _DTYPE_MAX):
                raise ValueError(
                    f"Sequence {index} contains a value outside uint16 range 0..{_DTYPE_MAX}."
                )
            arr = values.astype(_DTYPE, copy=False)
            data_file.write(arr.tobytes(order="C"))
            sizes.append(len(seq))

    # Build pointers (byte offset of each item in .bin)
    dtype_size = _DTYPE().itemsize
    pointers = []
    addr = 0
    for size in sizes:
        pointers.append(addr)
        addr += size * dtype_size

    with open(idx_path, "wb") as idx_file:
        idx_file.write(_HDR_MAGIC)
        idx_file.write(struct.pack("<Q", _VERSION))
        idx_file.write(struct.pack("<B", _DTYPE_CODE))
        idx_file.write(struct.pack("<Q", len(sizes)))
        idx_file.write(np.array(sizes, dtype=np.int32).tobytes(order="C"))
        idx_file.write(np.array(pointers, dtype=np.int64).tobytes(order="C"))
