"""Pure Python MXNet RecordIO reader — no mxnet dependency."""

import struct
import io
import os
from PIL import Image
import numpy as np

kMagic = 0xced7230a

# 解码后端: pil / turbojpeg / torchvision / cv2 / turbojpeg_tensor
# turbojpeg 默认: 真实7卡DDP训练实测比 PIL 快 10% (2.90 vs 3.23 min/1000batch)
# turbojpeg_tensor: 跳过 PIL, 直接返回归一化后的 tensor (需要 augmenter 也支持 tensor)
_DECODE_BACKEND = os.environ.get('DECODE_BACKEND', 'turbojpeg').lower()
_TJ_INSTANCE = None


def _get_turbojpeg():
    global _TJ_INSTANCE
    if _TJ_INSTANCE is None:
        from turbojpeg import TurboJPEG
        _TJ_INSTANCE = TurboJPEG()
    return _TJ_INSTANCE


class RecordIOReader:
    """Read MXNet RecordIO (.rec/.idx) files without mxnet.

    Fork-safe: detects PID changes and reopens file handles in child processes.
    """

    def __init__(self, idx_path, rec_path):
        self.offsets = {}
        with open(idx_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    self.offsets[int(parts[0])] = int(parts[1])
        self.rec_path = rec_path
        self._pid = os.getpid()
        self.rec_file = open(rec_path, 'rb')

    def _check_pid(self):
        """Reopen file handle if we're in a forked child process."""
        if os.getpid() != self._pid:
            self._pid = os.getpid()
            self.rec_file = open(self.rec_path, 'rb')

    @property
    def keys(self):
        return self.offsets.keys()

    def read_idx(self, idx):
        self._check_pid()
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
        """Decode image bytes to PIL Image in RGB.

        Backend controlled by DECODE_BACKEND env var:
          pil (default), turbojpeg, torchvision, cv2
        """
        if _DECODE_BACKEND == 'turbojpeg':
            from turbojpeg import TJPF_RGB
            tj = _get_turbojpeg()
            img_np = tj.decode(img_bytes, pixel_format=TJPF_RGB)
            return Image.fromarray(img_np)
        elif _DECODE_BACKEND == 'torchvision':
            import torch
            import torchvision.io
            data = torch.frombuffer(bytearray(img_bytes), dtype=torch.uint8)
            img_tensor = torchvision.io.decode_jpeg(data)  # (3,H,W) uint8
            img_np = img_tensor.permute(1, 2, 0).numpy()
            return Image.fromarray(img_np)
        elif _DECODE_BACKEND == 'cv2':
            import cv2
            buf = np.frombuffer(img_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img_rgb)
        else:
            # PIL (default)
            img = Image.open(io.BytesIO(img_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img

    @staticmethod
    def decode_image_numpy(img_bytes):
        """Decode image bytes directly to numpy array (H, W, 3) uint8 RGB.

        Skips PIL entirely — for use with v2 augmenters.
        """
        if _DECODE_BACKEND == 'turbojpeg':
            from turbojpeg import TJPF_RGB
            tj = _get_turbojpeg()
            return tj.decode(img_bytes, pixel_format=TJPF_RGB)
        elif _DECODE_BACKEND == 'cv2':
            import cv2
            buf = np.frombuffer(img_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif _DECODE_BACKEND == 'torchvision':
            import torch
            import torchvision.io
            data = torch.frombuffer(bytearray(img_bytes), dtype=torch.uint8)
            img_tensor = torchvision.io.decode_jpeg(data)  # (3,H,W)
            return img_tensor.permute(1, 2, 0).numpy()
        else:
            # PIL fallback → numpy
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
