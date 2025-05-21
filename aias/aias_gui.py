import os
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QTextEdit, QMessageBox, QListWidget, QHBoxLayout,
    QDialog, QLabel
)
from PyQt5.QtCore import QTimer

# Ensure project root in path
this_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(this_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from aias.agent import (
    index_files, handle_input,
    get_pending_patches, background_tasks,
    propose_patch
)

class PatchReviewDialog(QDialog):
    def __init__(self, filename, description, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Review Patch: {filename}")
        self.setMinimumSize(500, 300)

        layout = QVBoxLayout(self)

        label = QLabel(f"<b>{filename}</b>")
        layout.addWidget(label)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setText(description)
        layout.addWidget(self.text)

        approve_btn = QPushButton("✅ Approve Patch")
        decline_btn = QPushButton("❌ Decline Patch")

        approve_btn.clicked.connect(lambda: self.accept_patch(filename, description))
        decline_btn.clicked.connect(self.reject)

        layout.addWidget(approve_btn)
        layout.addWidget(decline_btn)

    def accept_patch(self, fn, desc):
        propose_patch(fn, desc)
        self.accept()

class GuiMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AIAS GUI")
        self.resize(600, 500)

        index_files(project_root)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.chatbox = QTextEdit()
        self.chatbox.setReadOnly(True)
        layout.addWidget(self.chatbox)

        self.input_box = QTextEdit()
        self.input_box.setFixedHeight(60)
        layout.addWidget(self.input_box)

        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self.on_send)
        layout.addWidget(send_btn)

        self.patch_list = QListWidget()
        self.patch_list.itemDoubleClicked.connect(self.on_patch_selected)
        layout.addWidget(self.patch_list)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_patches)
        self.timer.start(1000)

    def on_send(self):
        user_text = self.input_box.toPlainText().strip()
        if not user_text:
            return
        self.chatbox.append(f"User: {user_text}")
        response = handle_input(user_text)
        for line in response.splitlines():
            self.chatbox.append(f"AIAS: {line}")
        self.input_box.clear()

    def refresh_patches(self):
        self.patch_list.clear()
        for fn, desc in get_pending_patches():
            title = f"{fn}: {desc[:80].strip()}..."
            self.patch_list.addItem(title)

    def on_patch_selected(self, item):
        if not item:
            return
        fn, desc = item.text().split(":", 1)
        dialog = PatchReviewDialog(fn.strip(), desc.strip(), self)
        dialog.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GuiMainWindow()
    window.show()
    sys.exit(app.exec_())
