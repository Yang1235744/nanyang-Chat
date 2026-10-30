"""桌面端模型引擎：聊天 LLM + Helios 图/视频生成，全部本地推理、吃本机硬件。

权重查找顺序（都找不到则对应功能显示"未训练"）：
  聊天：环境变量 CHAT_CKPT > %APPDATA%/nanyang-Chat/models/chat_latest.pt > 项目 runs/chat_tiny/latest.pt
  Helios：环境变量 HELIOS_CKPT > %APPDATA%/nanyang-Chat/models/helios_latest_dit.pt > Helios 项目默认路径
"""

import os
import sys
import tempfile
import uuid

import torch
import yaml

# 冻结成 EXE 时资源在 _MEIPASS；helios 包由 PyInstaller 直接打进软件
BASE = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

APP_DATA = os.path.join(os.environ.get("APPDATA", tempfile.gettempdir()), "nanyang-Chat")
OUT_DIR = os.path.join(APP_DATA, "outputs")
MODELS_DIR = os.path.join(APP_DATA, "models")
try:
    for d in (APP_DATA, OUT_DIR, MODELS_DIR):
        os.makedirs(d, exist_ok=True)
    # 验证真的可写（某些沙箱环境 Roaming 不可写）
    probe = os.path.join(APP_DATA, ".probe")
    with open(probe, "w") as fh:
        fh.write("ok")
    os.remove(probe)
except OSError:  # 回退到临时目录，功能不受影响
    APP_DATA = os.path.join(tempfile.gettempdir(), "nanyang-Chat")
    OUT_DIR = os.path.join(APP_DATA, "outputs")
    MODELS_DIR = os.path.join(APP_DATA, "models")
    for d in (APP_DATA, OUT_DIR, MODELS_DIR):
        os.makedirs(d, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open(os.path.join(BASE, "configs", "chat_tiny.yaml"), "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

NOT_TRAINED = (
    "本地聊天大模型还没训练。把语料 .txt 放进 data/corpus/ 后运行：\n"
    "python -m nchat.train --config configs/chat_tiny.yaml\n"
    f"把得到的 latest.pt 放到 {MODELS_DIR} 下改名 chat_latest.pt 即可启用对话。"
)


def _pick(env_key: str, *candidates: str) -> str:
    if os.environ.get(env_key):
        return os.environ[env_key]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return candidates[0] if candidates else ""


class ChatLLM:
    def __init__(self):
        from nchat.models import TinyGPT

        path = _pick("CHAT_CKPT",
                     os.path.join(MODELS_DIR, "chat_latest.pt"),
                     os.path.join(BASE, "runs", "chat_tiny", "latest.pt"))
        self.model = None
        if os.path.isfile(path):
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            cfg = ckpt.get("config", {})
            self.model = TinyGPT(**cfg.get("gpt", CFG["gpt"])).to(DEVICE).eval()
            self.model.load_state_dict(ckpt["gpt"])

    @property
    def status(self) -> str:
        return "ready" if self.model is not None else "untrained"

    def reply(self, messages: list) -> str:
        if self.model is None:
            return NOT_TRAINED
        from nchat.generate import reply as gpt_reply
        return gpt_reply(self.model, messages, device=DEVICE, **CFG["gen"])


class HeliosBridge:
    def __init__(self):
        self.ok = False
        self.err = None
        try:
            try:
                from helios.diffusion.flow import euler_sample  # EXE 内嵌
                from helios.models import ByteTextEncoder, HeliosDiT, TinyVAE
                from helios.models.text_encoder import PAD_ID, encode_text
            except ImportError:  # 开发环境：从兄弟项目导入
                root = os.environ.get("HELIOS_ROOT", os.path.normpath(
                    os.path.join(BASE, "..", "nanyang-Helios1.2")))
                if root not in sys.path:
                    sys.path.insert(0, root)
                from helios.diffusion.flow import euler_sample
                from helios.models import ByteTextEncoder, HeliosDiT, TinyVAE
                from helios.models.text_encoder import PAD_ID, encode_text
            self._euler, self._encode_text, self._PAD = euler_sample, encode_text, PAD_ID

            path = _pick("HELIOS_CKPT",
                         os.path.join(MODELS_DIR, "helios_latest_dit.pt"),
                         os.path.join(BASE, "..", "nanyang-Helios1.2", "runs", "helios_tiny", "latest_dit.pt"))
            if not os.path.isfile(path):
                self.err = f"未找到 Helios 权重（把 latest_dit.pt 放到 {MODELS_DIR} 改名 helios_latest_dit.pt）"
                return
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            self.cfg = ckpt["config"]
            assert ckpt.get("dit") is not None, "checkpoint 中没有 DiT 权重"
            self.vae = TinyVAE(**self.cfg["vae"]).to(DEVICE).eval().requires_grad_(False)
            self.text = ByteTextEncoder(**self.cfg["text"]).to(DEVICE).eval()
            self.dit = HeliosDiT(text_dim=self.cfg["text"]["dim"], **self.cfg["dit"]).to(DEVICE).eval()
            self.vae.load_state_dict(ckpt["vae"])
            self.text.load_state_dict(ckpt["text"])
            self.dit.load_state_dict(ckpt["dit"])
            self.ok = True
        except Exception as e:  # 桥接失败不影响聊天
            self.err = f"Helios 加载失败: {e}"

    @property
    def status(self) -> str:
        return "ready" if self.ok else "no_checkpoint"

    @torch.no_grad()
    def generate(self, prompt: str, kind: str) -> str:
        if not self.ok:
            raise RuntimeError(self.err)
        import imageio.v3 as iio
        from PIL import Image

        cfg = self.cfg
        max_len = cfg["text"]["max_len"]
        size = cfg["train"]["image_size"]
        frames = 1 if kind == "image" else cfg["train"]["num_frames"]

        ids_c = torch.tensor([self._encode_text(prompt, max_len)], device=DEVICE)
        ids_u = torch.tensor([self._encode_text("", max_len)], device=DEVICE)
        ctx_c, ctx_u = self.text(ids_c), self.text(ids_u)
        h = w = size // self.vae.downsample_factor
        z = self._euler(self.dit, ctx_c, ids_c == self._PAD, ctx_u, ids_u == self._PAD,
                        shape=(1, cfg["vae"]["latent_channels"], frames, h, w),
                        steps=cfg["sample"]["steps"], cfg_scale=cfg["sample"]["cfg_scale"], device=DEVICE)
        z = z / cfg["latent_scale"]
        px = self.vae.decode(z.permute(0, 2, 1, 3, 4).reshape(frames, -1, h, w))
        arr = ((px.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

        path = os.path.join(OUT_DIR, f"{uuid.uuid4().hex[:12]}.{'png' if kind == 'image' else 'gif'}")
        if kind == "image":
            Image.fromarray(arr[0]).save(path)
        else:
            iio.imwrite(path, arr, duration=int(1000 / cfg["sample"]["fps"]), loop=0)
        return path


class Engine:
    """UI 只跟它打交道。"""

    def __init__(self):
        self.llm = ChatLLM()
        self.helios = HeliosBridge()

    def status(self) -> dict:
        return {"device": DEVICE, "llm": self.llm.status, "helios": self.helios.status,
                "helios_error": self.helios.err}

    def chat(self, messages: list) -> str:
        return self.llm.reply(messages)

    def generate(self, prompt: str, kind: str) -> str:
        return self.helios.generate(prompt, kind)
