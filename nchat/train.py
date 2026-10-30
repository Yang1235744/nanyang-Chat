"""训练入口（在项目根目录运行）。

语料：data/corpus/*.txt（对话格式：用户：...\\n助手：...）
运行：python -m nchat.train --config configs/chat_tiny.yaml
"""

import argparse
import os
import tempfile

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from nchat.data import TextCorpus
from nchat.models import TinyGPT


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr(step, base, warmup, total):
    import math
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    t = cfg["train"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    ds = TextCorpus(t["data_dir"], t["block_size"])
    loader = DataLoader(ds, batch_size=t["batch_size"], shuffle=True, drop_last=True)
    model = TinyGPT(**cfg["gpt"]).to(device)
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"])
    print(f"[{cfg['model_name']}] tokens={len(ds.ids)} params={sum(p.numel() for p in params) / 1e6:.1f}M device={device}")

    start = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["gpt"])
        start = ckpt.get("step", 0)

    out_dir = t["out_dir"]
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError:  # 工作区盘只读时退回临时目录
        out_dir = os.path.join(tempfile.gettempdir(), "nanyang-Chat", "runs")
        os.makedirs(out_dir, exist_ok=True)
        print(f"输出目录不可写，改用 {out_dir}")

    it = iter(loader)
    pbar = tqdm(range(start, t["steps"]), initial=start, total=t["steps"])
    for step in pbar:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        lr = get_lr(step, t["lr"], t["warmup_steps"], t["steps"])
        for g in opt.param_groups:
            g["lr"] = lr

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, t["grad_clip"])
        opt.step()

        if (step + 1) % t["log_every"] == 0:
            pbar.set_postfix_str(f"loss={loss.item():.4f} lr={lr:.2e}")
        if (step + 1) % t["save_every"] == 0 or (step + 1) == t["steps"]:
            state = {"step": step + 1, "config": cfg, "gpt": model.state_dict()}
            torch.save(state, os.path.join(out_dir, f"ckpt_{step + 1}.pt"))
            torch.save(state, os.path.join(out_dir, "latest.pt"))

    print(f"训练完成，checkpoint 在 {out_dir}")


if __name__ == "__main__":
    main()
