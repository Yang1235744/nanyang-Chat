"""对话生成：纯文本对话模板（用户：/助手：），生成后按 "\\n用户：" 截断。"""

import torch

from .tokenizer import BOS_ID, EOS_ID, decode, encode

USER, BOT = "用户：", "助手："


def build_prompt(messages: list[dict], history: int = 5) -> str:
    """messages: OpenAI 风格 [{role, content}]，取最近 history 轮 + 当前问题。"""
    pairs, pending = [], None
    for m in messages:
        if m["role"] == "user":
            pending = m["content"]
        elif m["role"] == "assistant" and pending is not None:
            pairs.append((pending, m["content"]))
            pending = None
    s = ""
    for q, a in pairs[-history:]:
        s += f"{USER}{q}\n{BOT}{a}\n"
    if pending is not None:
        s += f"{USER}{pending}\n{BOT}"
    return s


@torch.no_grad()
def reply(model, messages: list[dict], max_new_tokens: int = 200, temperature: float = 0.8,
          top_k: int = 40, history: int = 5, device: str = "cpu") -> str:
    prompt = build_prompt(messages, history)
    ids = torch.tensor([[BOS_ID] + encode(prompt)], device=device)
    out = model.generate(ids, max_new=max_new_tokens, temperature=temperature, top_k=top_k)[0].tolist()
    if EOS_ID in out:
        out = out[: out.index(EOS_ID)]
    text = decode(out[len(ids[0]):])
    return text.split("\n" + USER)[0].strip()
