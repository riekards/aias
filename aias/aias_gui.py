# aias/aias_gui.py

import os
import sys
import re
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QTextEdit, QMessageBox, QListWidget, QHBoxLayout
)
from PyQt5.QtCore import QTimer

# ensure project root in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from aias.agent import handle_input, background_tasks, completed_tasks, _propose_and_save_patch, resolve_path

class GuiMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AIAS GUI")
        self.resize(700, 600)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Chat display
        self.chatbox = QTextEdit()
        self.chatbox.setReadOnly(True)
        layout.addWidget(self.chatbox)

        # Input box
        self.input_box = QTextEdit()
        self.input_box.setFixedHeight(60)
        layout.addWidget(self.input_box)

        # Send button
        self.exec_btn = QPushButton("Send")
        self.exec_btn.clicked.connect(self.on_send)
        layout.addWidget(self.exec_btn)

        # Patch request list
        self.patch_list = QListWidget()
        layout.addWidget(self.patch_list)

        # Approve/Decline row
        btn_row = QHBoxLayout()
        self.approve_btn = QPushButton("Apply Selected Patch")
        self.decline_btn = QPushButton("Decline Selected Patch")
        btn_row.addWidget(self.approve_btn)
        btn_row.addWidget(self.decline_btn)
        layout.addLayout(btn_row)

        self.approve_btn.clicked.connect(self.on_approve)
        self.decline_btn.clicked.connect(self.on_decline)

        # Timer to refresh patch list every second
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_patches)
        self.timer.start(1000)

    def on_send(self):
        user_text = self.input_box.toPlainText().strip()
        if not user_text:
            return
        # display user
        self.chatbox.append(f"<b>You:</b> {user_text}")
        # process input
        try:
            ai_reply = handle_input(user_text)
        except Exception as e:
            ai_reply = f"⚠️ Exception in handle_input: {e}"
        # display AI
        for line in ai_reply.splitlines():
            self.chatbox.append(f"<b>AIAS:</b> {line}")
        self.input_box.clear()

    def refresh_patches(self):
        """
        Show any pending background tasks that have completed.
        """
        self.patch_list.clear()
        for fn, desc in completed_tasks:
            self.patch_list.addItem(f"{fn}: {desc}")

    def on_approve(self):
        """
        Apply the selected patch immediately.
        """
        item = self.patch_list.currentItem()
        if not item:
            return
        text = item.text()
        fn, desc = text.split(":", 1)
        fn = fn.strip()
        desc = desc.strip()
        # confirm with user
        resp = QMessageBox.question(
            self, "Apply Patch",
            f"Apply this patch to {fn}?\n\n{desc}",
            QMessageBox.Yes | QMessageBox.No
        )
        if resp == QMessageBox.Yes:
            try:
                # call the same internal function as background worker
                _propose_and_save_patch(fn, desc)
                QMessageBox.information(self, "Patch Applied", f"Applied patch to {fn}")
                # remove from list
                completed_tasks[:] = [(f,d) for f,d in completed_tasks if not (f==fn and d==desc)]
                self.refresh_patches()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to apply patch: {e}")

    def on_decline(self):
        """
        Decline (remove) the selected patch.
        """
        item = self.patch_list.currentItem()
        if not item:
            return
        text = item.text()
        fn, desc = text.split(":", 1)
        fn = fn.strip()
        desc = desc.strip()
        # remove from completed_tasks
        before = len(completed_tasks)
        completed_tasks[:] = [(f,d) for f,d in completed_tasks if not (f==fn and d==desc)]
        after = len(completed_tasks)
        if before != after:
            QMessageBox.information(self, "Patch Declined", f"Declined patch for {fn}")
            self.refresh_patches()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = GuiMainWindow()
    win.show()
    sys.exit(app.exec_())
