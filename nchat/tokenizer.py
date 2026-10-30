"""字节级 tokenizer —— 与 nanyang-Helios1.2 文本编码器同一套约定。

词表 = 256 个 UTF-8 字节 + PAD/BOS/EOS，中英文原生支持，零外部依赖。
对话格式直接用纯文本标记："用户：...\\n助手：..."，不需要特殊 token。
"""

PAD_ID, BOS_ID, EOS_ID = 256, 257, 258
VOCAB_SIZE = 259


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(ids) -> str:
    return bytes(i for i in ids if isinstance(i, int) and 0 <= i < 256).decode("utf-8", errors="replace")
