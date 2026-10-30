"""语料数据集：目录下所有 .txt 拼成字节 token 流，按 block_size 切窗。"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import EOS_ID, PAD_ID, encode


class TextCorpus(Dataset):
    def __init__(self, root: str, block_size: int):
        ids = []
        files = []
        for dirpath, _, names in os.walk(root):
            files += [os.path.join(dirpath, n) for n in sorted(names) if n.endswith(".txt")]
        if not files:
            raise RuntimeError(f"语料目录为空: {root}，请放入 .txt 语料文件")
        for f in files:
            with open(f, "r", encoding="utf-8") as fh:
                ids.extend(encode(fh.read()))
            ids.append(EOS_ID)
        self.ids = np.array(ids, dtype=np.uint16)
        self.block = block_size

    def __len__(self):
        return max(1, (len(self.ids) - 1) // self.block)

    def __getitem__(self, i):
        lo = i * self.block
        x = np.full(self.block, PAD_ID, dtype=np.int64)
        y = np.full(self.block, PAD_ID, dtype=np.int64)
        seg = self.ids[lo:lo + self.block + 1].astype(np.int64)
        x[: len(seg) - 1] = seg[:-1]
        y[: len(seg) - 1] = seg[1:]
        return torch.from_numpy(x), torch.from_numpy(y)
