"""冒烟测试：GPT + 引擎加载（未训练兜底）+ 离屏创建主窗口。

运行（项目根目录）：python scripts/smoke_test.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import torch


def main():
    torch.manual_seed(0)

    from nchat.models import TinyGPT
    gpt = TinyGPT(dim=64, depth=2, heads=2, max_len=32)
    ids = torch.randint(0, 259, (2, 16))
    _, loss = gpt(ids, ids)
    loss.backward()
    print(f"[1] TinyGPT ok, loss={loss.item():.4f}")

    from nchat.generate import reply
    text = reply(gpt, [{"role": "user", "content": "你好"}], max_new_tokens=5, temperature=0)
    print(f"[2] generate ok, reply={text[:20]!r}")

    from desktop.engine import Engine
    eng = Engine()
    s = eng.status()
    print(f"[3] engine ok: llm={s['llm']} helios={s['helios']} device={s['device']}")
    r = eng.chat([{"role": "user", "content": "你好"}])
    assert r
    print(f"[4] engine.chat ok: {r[:24]!r}…")

    from PySide6.QtWidgets import QApplication
    from desktop.main_window import MainWindow
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.new_conv()
    print("[5] MainWindow 离屏创建 + 新会话 ok")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
