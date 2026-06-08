import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)
import os, sys

sys.path.append(os.path.join(root))

import re
import json
import struct
import argparse
import numpy as np
import pandas as pd
import lmdb
from io import BytesIO
from tqdm import tqdm
from PIL import Image


# ===================== 工具函数 =====================

def natural_sort(l):
    """自然排序：file2 排在 file10 前面"""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', str(key))]
    return sorted(l, key=alphanum_key)


def get_all_files(root_dir, extension_list=None, sort=False):
    """递归获取目录下所有指定扩展名的文件路径"""
    all_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        all_files += [os.path.join(dirpath, f) for f in filenames]
    if sort:
        all_files = natural_sort(all_files)
    if extension_list is not None:
        all_files = [f for f in all_files if os.path.splitext(f)[1].lower() in extension_list]
    return all_files


# ===================== LMDB Writer =====================

class LMDBWriter:
    """
    将图片打包写入 LMDB 数据库。

    LMDB 存储结构：
      - key: 图片索引（8位零填充字符串，如 b'00000000', b'00000001', ...）
      - value: [4 bytes: label (int32 LE)] [N bytes: JPEG image data]

    额外的元数据 key：
      - b'__len__'        : 数据库中图片的总数量（UTF-8 字符串）
      - b'__num_classes__' : 唯一标签的数量（UTF-8 字符串）
      - b'__label_map__'  : label_name -> label_id 的 JSON 映射
    """

    LABEL_BYTES = 4  # int32

    def __init__(self, save_root, prefix='train', map_size=1 << 40):
        self.save_root = save_root
        self.prefix = prefix
        os.makedirs(save_root, exist_ok=True)

        self.lmdb_path = os.path.join(save_root, f'{prefix}.lmdb')
        self.list_path = os.path.join(save_root, f'{prefix}.tsv')
        self.done_path = os.path.join(save_root, 'done_list.txt')

        # 清理已有文件
        for path in [self.list_path, self.done_path]:
            if os.path.isfile(path):
                os.remove(path)
        if os.path.isdir(self.lmdb_path):
            import shutil
            shutil.rmtree(self.lmdb_path)

        self.env = lmdb.open(self.lmdb_path, map_size=map_size)
        self.list_writer = open(self.list_path, 'w')
        self.done_writer = open(self.done_path, 'w')
        self.image_index = 0
        self.txn = self.env.begin(write=True)
        self._commit_interval = 5000

    @staticmethod
    def _encode_key(index: int) -> bytes:
        return f'{index:08d}'.encode('ascii')

    @staticmethod
    def _pack_value(label: int, img_bytes: bytes) -> bytes:
        return struct.pack('<i', label) + img_bytes

    @staticmethod
    def unpack_value(raw: bytes):
        label = struct.unpack('<i', raw[:4])[0]
        img_bytes = raw[4:]
        return label, img_bytes

    def write(self, rgb_pil_img, save_path: str, label: int, quality: int = 100):
        assert isinstance(label, int), f"label 必须是 int，收到 {type(label)}"

        buf = BytesIO()
        img = rgb_pil_img.convert('RGB') if rgb_pil_img.mode != 'RGB' else rgb_pil_img
        img.save(buf, format='JPEG', quality=quality)
        img_bytes = buf.getvalue()

        key = self._encode_key(self.image_index)
        value = self._pack_value(label, img_bytes)
        self.txn.put(key, value)

        line = f'{self.image_index}\t{save_path}\t{label}\n'
        self.list_writer.write(line)

        self.image_index += 1

        if self.image_index % self._commit_interval == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)

    def mark_done(self, context, name):
        line = f'{context}\t{name}\n'
        self.done_writer.write(line)

    def close(self, num_classes: int = 0, label_mapping: dict = None):
        self.txn.put(b'__len__', str(self.image_index).encode('utf-8'))
        self.txn.put(b'__num_classes__', str(num_classes).encode('utf-8'))
        if label_mapping is not None:
            self.txn.put(b'__label_map__', json.dumps(label_mapping, ensure_ascii=False).encode('utf-8'))

        self.txn.commit()
        self.env.close()
        self.list_writer.close()
        self.done_writer.close()
        print(f'LMDB 已保存到: {self.lmdb_path}')
        print(f'  总图片数: {self.image_index}')
        print(f'  类别数:   {num_classes}')


# ===================== LMDB Reader =====================

class LMDBReader:
    """
    读取由 LMDBWriter 创建的 LMDB 数据库。

    用法：
        reader = LMDBReader('/path/to/save_dir', prefix='train')
        label, pil_img = reader[0]
        print(len(reader), reader.num_classes)
        reader.close()
    """

    def __init__(self, db_path_or_dir, prefix='train'):
        if os.path.isdir(os.path.join(db_path_or_dir, f'{prefix}.lmdb')):
            self.lmdb_path = os.path.join(db_path_or_dir, f'{prefix}.lmdb')
        else:
            self.lmdb_path = db_path_or_dir

        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False,
                             readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            self._length = int(txn.get(b'__len__').decode('utf-8'))
            num_cls_raw = txn.get(b'__num_classes__')
            self._num_classes = int(num_cls_raw.decode('utf-8')) if num_cls_raw else 0
            label_map_raw = txn.get(b'__label_map__')
            self._label_map = json.loads(label_map_raw.decode('utf-8')) if label_map_raw else None

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def label_map(self):
        return self._label_map

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        key = f'{index:08d}'.encode('ascii')
        with self.env.begin(write=False) as txn:
            raw = txn.get(key)
        if raw is None:
            raise IndexError(f'索引 {index} 不在数据库中')
        label, img_bytes = LMDBWriter.unpack_value(raw)
        img = Image.open(BytesIO(img_bytes)).convert('RGB')
        return label, img

    def close(self):
        self.env.close()


# ===================== 主程序 =====================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='将图片打包为 LMDB 格式。图片应存储在以标签命名的子目录中，'
                    '例如 label1/image1.png, label2/image2.png。')
    parser.add_argument('--source_dir', type=str, required=True,
                        help='包含标签子目录的图片源目录')
    parser.add_argument('--save_dir', default='', type=str,
                        help='LMDB 保存目录，默认与 source_dir 相同')
    parser.add_argument('--quality', type=int, default=100,
                        help='JPEG 压缩质量 (1-100)，默认 100')
    parser.add_argument('--map_size', type=int, default=1099511627776,
                        help='LMDB 最大容量（字节），默认 1TB')
    parser.add_argument('--remove_images', action='store_true',
                        help='打包完成后删除源图片目录')

    args = parser.parse_args()
    source_dir = args.source_dir.rstrip('/')

    if not args.save_dir:
        save_dir = source_dir
    else:
        save_dir = args.save_dir
        os.makedirs(save_dir, exist_ok=True)

    # 查找所有图片
    print('正在查找目录中的所有图片:', source_dir)
    all_image_paths = get_all_files(source_dir, extension_list=['.jpg', '.png', '.jpeg', '.bmp', '.webp'], sort=True)
    print(f'共找到 {len(all_image_paths)} 张图片，位于 {source_dir}')

    if len(all_image_paths) == 0:
        print('未找到任何图片，退出。')
        sys.exit(1)

    # 从目录结构解析标签
    paths = pd.Series(all_image_paths)
    dataset = pd.DataFrame(paths, columns=['path'])
    dataset['rel_path'] = dataset['path'].apply(lambda x: x.replace(source_dir + '/', ''))
    dataset['label'] = dataset['rel_path'].apply(lambda x: x.split('/')[0])
    dataset['image_name'] = dataset['rel_path'].apply(lambda x: '/'.join(x.split('/')[1:]))

    unique_subject_ids = natural_sort(dataset['label'].unique().tolist())
    num_classes = len(unique_subject_ids)
    print(f'唯一标签数: {num_classes}')

    label_mapping = {sid: idx for idx, sid in enumerate(unique_subject_ids)}

    # 创建 LMDB Writer
    writer = LMDBWriter(save_dir, prefix='train', map_size=args.map_size)

    num_done = -1
    for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc='写入图片到 LMDB'):
        try:
            orig_rgb_pil_img = Image.open(row['path']).convert('RGB')
            label = label_mapping[row['label']]
            save_path = f'{label}/{row["image_name"]}'
            writer.write(rgb_pil_img=orig_rgb_pil_img, save_path=save_path, label=label, quality=args.quality)
            writer.mark_done(i, save_path)
            num_done += 1
            if num_done % 10000 == 0:
                os.makedirs(os.path.join(save_dir, 'examples'), exist_ok=True)
                orig_rgb_pil_img.save(os.path.join(save_dir, 'examples', f'{num_done}.jpg'))
        except Exception as e:
            print(f"{row['path']} {row['image_name']} 图片读取出错: {e}")
            sys.exit(1)

    writer.close(num_classes=num_classes, label_mapping=label_mapping)

    # 验证
    print('\n正在验证 LMDB 数据...')
    reader = LMDBReader(save_dir, prefix='train')
    print(f'  LMDB 总记录数: {len(reader)}')
    print(f'  LMDB 类别数:   {reader.num_classes}')
    check_indices = [0, len(reader) // 2, len(reader) - 1]
    for idx in check_indices:
        label, img = reader[idx]
        print(f'  索引 {idx}: label={label}, size={img.size}, mode={img.mode}')
    reader.close()
    print('验证完成，LMDB 数据正常。')

    # 删除源图片目录
    if args.remove_images:
        import shutil
        for d in os.listdir(source_dir):
            full_path = os.path.join(source_dir, d)
            if os.path.isdir(full_path) and d != 'examples' and not d.endswith('.lmdb'):
                shutil.rmtree(full_path)
        print('源图片目录已清理。')
