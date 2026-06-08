"""Pure Python MXNet RecordIO reader — no mxnet dependency."""

import struct
import io
from PIL import Image
import numpy as np

kMagic = 0xced7230a


class RecordIOReader:
    """Read MXNet RecordIO (.rec/.idx) files without mxnet."""

    def __init__(self, idx_path, rec_path):
        self.offsets = {}
        with open(idx_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    self.offsets[int(parts[0])] = int(parts[1])
        self.rec_file = open(rec_path, 'rb')

    @property
    def keys(self):
        return self.offsets.keys()

    def read_idx(self, idx):
        self.rec_file.seek(self.offsets[idx])
        # Read magic + lrecord
        header_bytes = self.rec_file.read(8)
        if len(header_bytes) < 8:
            raise EOFError(f"Unexpected end of file at index {idx}")
        magic, lrecord = struct.unpack('<II', header_bytes)
        assert magic == kMagic, f"Invalid magic: {magic:#x}, expected {kMagic:#x}"

        cflag = (lrecord >> 29) & 7
        length = lrecord & ((1 << 29) - 1)
        data = self.rec_file.read(length)

        # Handle multi-part records (cflag: 1=start, 2=middle, 3=end)
        if cflag == 1:
            while True:
                # Skip padding to 4-byte boundary
                pad = (4 - (length % 4)) % 4
                if pad:
                    self.rec_file.read(pad)
                header_bytes = self.rec_file.read(8)
                magic2, lrecord2 = struct.unpack('<II', header_bytes)
                assert magic2 == kMagic
                cflag2 = (lrecord2 >> 29) & 7
                length2 = lrecord2 & ((1 << 29) - 1)
                data += self.rec_file.read(length2)
                if cflag2 == 3:  # end of multi-part
                    break

        return data

    @staticmethod
    def unpack(data):
        """Unpack a record into (header, image_bytes).

        Returns a (header, img_bytes) tuple compatible with mxnet's interface.
        header is a SimpleNamespace with .flag, .label, .id, .id2
        """
        flag, label_val, id_, id2 = struct.unpack('<IfQQ', data[:24])
        if flag == 0:
            label = label_val
            img_bytes = data[24:]
        else:
            # Multi-label: flag floats follow the 24-byte header
            label_end = 24 + flag * 4
            labels = struct.unpack(f'<{flag}f', data[24:label_end])
            label = labels
            img_bytes = data[label_end:]

        class Header:
            pass

        header = Header()
        header.flag = flag
        header.label = label
        header.id = id_
        header.id2 = id2
        return header, img_bytes

    @staticmethod
    def decode_image(img_bytes):
        """Decode image bytes to numpy array (H, W, C) in RGB."""
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)

    def close(self):
        self.rec_file.close()

    def __del__(self):
        try:
            self.rec_file.close()
        except Exception:
            pass
