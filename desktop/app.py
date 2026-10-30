"""应用入口：python run.py 启动桌面版。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from PySide6.QtWidgets import QApplication

    from desktop.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("nanyang-Chat")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
