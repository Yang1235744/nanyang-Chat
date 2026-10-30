"""后台线程：生成期间 UI 不卡死。"""

from PySide6.QtCore import QThread, Signal


class ChatWorker(QThread):
    done = Signal(str)

    def __init__(self, engine, messages):
        super().__init__()
        self._engine, self._messages = engine, messages

    def run(self):
        try:
            self.done.emit(self._engine.chat(self._messages))
        except Exception as e:
            self.done.emit(f"出错了：{e}")


class GenWorker(QThread):
    done = Signal(object)  # (ok, kind, path_or_error)

    def __init__(self, engine, prompt, kind):
        super().__init__()
        self._engine, self._prompt, self._kind = engine, prompt, kind

    def run(self):
        try:
            path = self._engine.generate(self._prompt, self._kind)
            self.done.emit((True, self._kind, path))
        except Exception as e:
            self.done.emit((False, self._kind, str(e)))
