"""主窗口：豆包式布局（侧栏会话 + 气泡消息流 + 底部输入），原生 Qt 控件。"""

import json
import os
import time

from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QMovie, QPixmap
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QMainWindow, QPlainTextEdit, QPushButton, QScrollArea,
                               QVBoxLayout, QWidget)

from desktop.engine import APP_DATA, Engine
from desktop.workers import ChatWorker, GenWorker

CONVS_FILE = os.path.join(APP_DATA, "conversations.json")

QSS = """
QMainWindow, QWidget { background: #0b0e14; color: #e8edf5; font-family: "Segoe UI", "Microsoft YaHei"; font-size: 14px; }
#sidebar { background: #121722; border-right: 1px solid #232b3a; }
#brand { font-family: "Cascadia Code", Consolas; font-size: 18px; font-weight: 700; padding: 6px; }
#newChat { background: #ff8a3d; color: #16100a; border-radius: 8px; padding: 10px; font-weight: 700; }
#newChat:hover { background: #ffa05c; }
QListWidget { background: transparent; border: 0; }
QListWidget::item { padding: 9px 12px; border-radius: 8px; color: #93a0b4; }
QListWidget::item:selected { background: #0d1117; color: #e8edf5; border-left: 2px solid #ff8a3d; }
QListWidget::item:hover:!selected { background: #0d1117; color: #e8edf5; }
#statusPill { font-family: Consolas; font-size: 11px; color: #93a0b4; border: 1px solid #232b3a; border-radius: 9px; padding: 2px 10px; }
#statusPill[ok="1"] { color: #52d273; border-color: #52d273; }
#statusPill[warn="1"] { color: #ff8a3d; border-color: #ff8a3d; }
#chatScroll { border: 0; }
#bubbleUser { background: #16233c; border: 1px solid #232b3a; border-radius: 12px; padding: 10px 14px; }
#bubbleBot { background: #121722; border: 1px solid #232b3a; border-radius: 12px; padding: 10px 14px; }
#avatarUser { background: #4da3ff; color: #0b1220; border-radius: 8px; font-weight: 700; }
#avatarBot { background: #ff8a3d; color: #16100a; border-radius: 8px; font-weight: 700; }
#meta { color: #93a0b4; font-size: 11px; font-family: Consolas; }
#typing { color: #93a0b4; font-family: Consolas; }
#composer { background: #121722; border-top: 1px solid #232b3a; }
.modeBtn { background: transparent; color: #93a0b4; border: 1px solid #232b3a; border-radius: 11px; padding: 4px 14px; font-family: Consolas; font-size: 12px; }
.modeBtn:checked { color: #ff8a3d; border-color: #ff8a3d; }
#input { background: #0d1117; border: 1px solid #232b3a; border-radius: 10px; padding: 10px 12px; }
#input:focus { border-color: #4da3ff; }
#send { background: #ff8a3d; color: #16100a; border-radius: 10px; padding: 0 22px; font-weight: 700; font-size: 15px; }
#send:disabled { background: #3a3f4d; color: #93a0b4; }
#emptyTitle { font-family: "Cascadia Code", Consolas; font-size: 30px; font-weight: 700; color: #ff8a3d; }
#emptySub { color: #93a0b4; }
"""


class Bubble(QFrame):
    """一条消息：头像 + 气泡（文本或图片/动图）。"""

    def __init__(self, msg, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(60, 6, 60, 6)
        avatar = QLabel("你" if msg["role"] == "user" else "H")
        avatar.setObjectName("avatarUser" if msg["role"] == "user" else "avatarBot")
        avatar.setFixedSize(32, 32)
        avatar.setAlignment(Qt.AlignCenter)

        box = QFrame()
        box.setObjectName("bubbleUser" if msg["role"] == "user" else "bubbleBot")
        bl = QVBoxLayout(box)
        bl.setContentsMargins(12, 10, 12, 10)
        if msg.get("kind") in ("image", "video") and msg.get("path"):
            meta = QLabel(("生成图片" if msg["kind"] == "image" else "生成视频") + " · " + msg.get("prompt", ""))
            meta.setObjectName("meta")
            bl.addWidget(meta)
            img = QLabel()
            if msg["path"].lower().endswith(".gif"):
                mv = QMovie(msg["path"])
                mv.setScaledSize(mv.scaledSize().scaled(360, 360, Qt.KeepAspectRatio))
                img.setMovie(mv)
                mv.start()
                img._movie = mv  # 防 GC
            else:
                img.setPixmap(QPixmap(msg["path"]).scaledToWidth(360, Qt.SmoothTransformation))
            bl.addWidget(img)
        else:
            lb = QLabel(msg.get("content", ""))
            lb.setWordWrap(True)
            lb.setTextInteractionFlags(Qt.TextSelectableByMouse)
            bl.addWidget(lb)
        box.setMaximumWidth(560)

        if msg["role"] == "user":
            row.addStretch(1)
            row.addWidget(box)
            row.addWidget(avatar, 0, Qt.AlignTop)
        else:
            row.addWidget(avatar, 0, Qt.AlignTop)
            row.addWidget(box)
            row.addStretch(1)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("nanyang Chat · 本地 AI 助手")
        self.resize(1080, 720)
        self.engine = Engine()
        self.mode = "chat"
        self.pending = False
        self._worker = None
        self._typing = None
        self.convs = self._load_convs()
        self.current = self.convs[0]["id"] if self.convs else None
        self._build_ui()
        self._refresh_status()
        self._reload_conv_list()
        self._reload_messages()

    # ---------- 数据 ----------
    def _load_convs(self):
        try:
            with open(CONVS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_convs(self):
        with open(CONVS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.convs, f, ensure_ascii=False, indent=1)

    def _conv(self):
        return next((c for c in self.convs if c["id"] == self.current), None)

    # ---------- UI ----------
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        lay = QHBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(240)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(14, 14, 14, 14)
        brand = QLabel("nanyang<font color='#ff8a3d'>/</font>Chat")
        brand.setObjectName("brand")
        sl.addWidget(brand)
        btn = QPushButton("＋ 新对话")
        btn.setObjectName("newChat")
        btn.clicked.connect(self.new_conv)
        sl.addWidget(btn)
        self.convList = QListWidget()
        self.convList.currentRowChanged.connect(self._on_conv_switch)
        sl.addWidget(self.convList, 1)
        self.stLlm = QLabel("LLM …")
        self.stHelios = QLabel("Helios …")
        for w in (self.stLlm, self.stHelios):
            w.setObjectName("statusPill")
            sl.addWidget(w)
        lay.addWidget(side)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        self.scroll = QScrollArea()
        self.scroll.setObjectName("chatScroll")
        self.scroll.setWidgetResizable(True)
        self.msgHost = QWidget()
        self.msgLay = QVBoxLayout(self.msgHost)
        self.msgLay.setContentsMargins(0, 12, 0, 12)
        self.msgLay.setSpacing(4)
        self.msgLay.addStretch(1)
        self.scroll.setWidget(self.msgHost)
        rl.addWidget(self.scroll, 1)

        composer = QFrame()
        composer.setObjectName("composer")
        cl = QVBoxLayout(composer)
        cl.setContentsMargins(60, 10, 60, 14)
        modes = QHBoxLayout()
        self.modeBtns = {}
        for key, label in (("chat", "聊天"), ("image", "生成图片"), ("video", "生成视频")):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setProperty("class", "modeBtn")
            b.setStyleSheet("")  # 走 QSS class 选择器
            b.setChecked(key == "chat")
            b.clicked.connect(lambda _, k=key: self._set_mode(k))
            self.modeBtns[key] = b
            modes.addWidget(b)
        modes.addStretch(1)
        cl.addLayout(modes)

        row = QHBoxLayout()
        self.input = QPlainTextEdit()
        self.input.setObjectName("input")
        self.input.setPlaceholderText("说点什么…（Enter 发送，Shift+Enter 换行）")
        self.input.setFixedHeight(64)
        self.input.installEventFilter(self)
        row.addWidget(self.input, 1)
        self.sendBtn = QPushButton("发送")
        self.sendBtn.setObjectName("send")
        self.sendBtn.setFixedHeight(64)
        self.sendBtn.clicked.connect(self.send)
        row.addWidget(self.sendBtn)
        cl.addLayout(row)
        rl.addWidget(composer)
        lay.addWidget(right, 1)
        self.setStyleSheet(QSS)
        # QSS 里 class 选择器对 modeBtn 需要手动设 objectName 循环，直接补样式
        for b in self.modeBtns.values():
            b.setObjectName("modeBtn")

    def eventFilter(self, obj, ev):
        if obj is self.input and ev.type() == QEvent.KeyPress:
            if ev.key() in (Qt.Key_Return, Qt.Key_Enter) and not (ev.modifiers() & Qt.ShiftModifier):
                self.send()
                return True
        return super().eventFilter(obj, ev)

    def _set_mode(self, key):
        self.mode = key
        for k, b in self.modeBtns.items():
            b.setChecked(k == key)
        self.input.setPlaceholderText(
            "说点什么…（Enter 发送，Shift+Enter 换行）" if key == "chat"
            else f"描述你想要的{'图片' if key == 'image' else '视频'}…")
        self.input.setFocus()

    def _refresh_status(self):
        s = self.engine.status()
        self.stLlm.setText("LLM " + ("就绪" if s["llm"] == "ready" else "未训练"))
        self.stLlm.setProperty("ok" if s["llm"] == "ready" else "warn", "1")
        self.stHelios.setText("Helios " + ("就绪" if s["helios"] == "ready" else "未训练"))
        self.stHelios.setProperty("ok" if s["helios"] == "ready" else "warn", "1")
        for w in (self.stLlm, self.stHelios):
            w.style().unpolish(w)
            w.style().polish(w)

    # ---------- 会话 ----------
    def new_conv(self):
        c = {"id": hex(int(time.time() * 1000))[2:], "title": "新对话", "messages": []}
        self.convs.insert(0, c)
        self.current = c["id"]
        self._save_convs()
        self._reload_conv_list()
        self._reload_messages()

    def _on_conv_switch(self, row):
        if 0 <= row < len(self.convs):
            self.current = self.convs[row]["id"]
            self._reload_messages()

    def _reload_conv_list(self):
        self.convList.blockSignals(True)
        self.convList.clear()
        for c in self.convs:
            self.convList.addItem(QListWidgetItem(c["title"][:20]))
        idx = next((i for i, c in enumerate(self.convs) if c["id"] == self.current), -1)
        if idx >= 0:
            self.convList.setCurrentRow(idx)
        self.convList.blockSignals(False)

    def _reload_messages(self):
        for i in reversed(range(self.msgLay.count() - 1)):
            w = self.msgLay.itemAt(i).widget()
            if w:
                w.deleteLater()
        c = self._conv()
        if not c or not c["messages"]:
            empty = QWidget()
            el = QVBoxLayout(empty)
            el.setAlignment(Qt.AlignCenter)
            t = QLabel("nanyang Chat")
            t.setObjectName("emptyTitle")
            t.setAlignment(Qt.AlignCenter)
            s = QLabel("本地运行的豆包式助手：聊天、生成图片、生成视频，无需登录")
            s.setObjectName("emptySub")
            s.setAlignment(Qt.AlignCenter)
            el.addWidget(t)
            el.addWidget(s)
            self.msgLay.insertWidget(0, empty)
            return
        for m in c["messages"]:
            self.msgLay.insertWidget(self.msgLay.count() - 1, Bubble(m))
        self._scroll_bottom()

    def _scroll_bottom(self):
        sb = self.scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ---------- 发送 ----------
    def send(self):
        text = self.input.toPlainText().strip()
        if not text or self.pending:
            return
        if not self._conv():
            self.new_conv()
        c = self._conv()
        if not c["messages"]:
            c["title"] = text[:18]
            self._reload_conv_list()
        c["messages"].append({"role": "user", "kind": "text", "content": text})
        self.input.clear()
        self._set_pending(True)
        self._reload_messages()

        if self.mode == "chat":
            history = [{"role": m["role"], "content": m["content"]}
                       for m in c["messages"] if m.get("kind") == "text"]
            self._worker = ChatWorker(self.engine, history)
            self._worker.done.connect(self._on_chat_done)
        else:
            self._worker = GenWorker(self.engine, text, self.mode)
            self._worker.done.connect(self._on_gen_done)
        self._worker.start()

    def _set_pending(self, on):
        self.pending = on
        self.sendBtn.setEnabled(not on)
        if on:
            self._typing = Bubble({"role": "assistant", "kind": "text",
                                   "content": "思考中…" if self.mode == "chat" else "生成中，扩散采样需要一点时间…"})
            self.msgLay.insertWidget(self.msgLay.count() - 1, self._typing)
            self._scroll_bottom()

    def _clear_pending(self):
        self.pending = False
        self.sendBtn.setEnabled(True)
        if self._typing:
            self._typing.deleteLater()
            self._typing = None

    def _append(self, msg):
        c = self._conv()
        c["messages"].append(msg)
        self._save_convs()
        self.msgLay.insertWidget(self.msgLay.count() - 1, Bubble(msg))
        self._scroll_bottom()

    def _on_chat_done(self, text):
        self._clear_pending()
        self._append({"role": "assistant", "kind": "text", "content": text})

    def _on_gen_done(self, result):
        self._clear_pending()
        ok, kind, payload = result
        if ok:
            self._append({"role": "assistant", "kind": kind, "path": payload, "prompt": ""})
        else:
            self._append({"role": "assistant", "kind": "text", "content": "生成失败：" + payload})
