import os
import sys
import json

# ── Make sure we can import the aias package ─────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QTextEdit, QMessageBox, QListWidget, QHBoxLayout,
    QLineEdit, QLabel, QSplitter
)
from PyQt5.QtCore import QTimer, Qt

from aias.agent import handle_input, background_tasks, completed_tasks, propose_patch, resolve_path


class GuiMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AIAS GUI")
        self.resize(800, 600)

        # Main splitter to divide chat and feature panel
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # Left side: Chat and patches
        left = QWidget()
        left_layout = QVBoxLayout(left)

        # Chat display
        self.chatbox = QTextEdit()
        self.chatbox.setReadOnly(True)
        left_layout.addWidget(self.chatbox)

        # Input box
        self.input_box = QLineEdit()
        self.input_box.returnPressed.connect(self.on_send)
        left_layout.addWidget(self.input_box)

        # Send button
        self.exec_btn = QPushButton("Send")
        self.exec_btn.clicked.connect(self.on_send)
        left_layout.addWidget(self.exec_btn)

        # Patch request list
        left_layout.addWidget(QLabel("Pending Patches:"))
        self.patch_list = QListWidget()
        left_layout.addWidget(self.patch_list)

        # Approve/Decline buttons
        btn_row = QHBoxLayout()
        self.approve_btn = QPushButton("Approve Patch")
        self.decline_btn = QPushButton("Decline Patch")
        btn_row.addWidget(self.approve_btn)
        btn_row.addWidget(self.decline_btn)
        left_layout.addLayout(btn_row)
        self.approve_btn.clicked.connect(self.on_approve)
        self.decline_btn.clicked.connect(self.on_decline)

        splitter.addWidget(left)

        # Right side: Feature requests panel
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("Feature Requests:"))

        # Feature input
        self.feature_input = QLineEdit()
        self.feature_input.setPlaceholderText("Describe new feature…")
        right_layout.addWidget(self.feature_input)

        self.add_feature_btn = QPushButton("Queue Feature Request")
        self.add_feature_btn.clicked.connect(self.on_queue_feature)
        right_layout.addWidget(self.add_feature_btn)

        # Feature queue list
        self.feature_list = QListWidget()
        right_layout.addWidget(self.feature_list)

        # Save feature requests button
        self.save_features_btn = QPushButton("Save Feature Requests")
        self.save_features_btn.clicked.connect(self.on_save_features)
        right_layout.addWidget(self.save_features_btn)

        splitter.addWidget(right)

        # Timer to refresh patch list every second
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_patches)
        self.timer.start(1000)

    def on_send(self):
        user_text = self.input_box.text().strip()
        if not user_text:
            return

        # Display user
        self.chatbox.append(f"You: {user_text}")

        # Handle via agent
        try:
            response = handle_input(user_text)
        except Exception as e:
            response = f"⚠️ Exception in handle_input: {e}"

        for line in response.splitlines():
            self.chatbox.append(f"AIAS: {line}")

        self.input_box.clear()

    def refresh_patches(self):
        self.patch_list.clear()
        for fn, desc in completed_tasks:
            self.patch_list.addItem(f"{fn}: {desc}")

    def on_approve(self):
        item = self.patch_list.currentItem()
        if not item:
            return
        fn, desc = item.text().split(":", 1)
        propose_patch(fn.strip(), desc.strip())
        QMessageBox.information(self, "Patch Applied", f"Applied patch to {fn.strip()}")

    def on_decline(self):
        item = self.patch_list.currentItem()
        if not item:
            return
        fn, desc = item.text().split(":", 1)
        # Remove from completed_tasks
        completed_tasks[:] = [
            (f, d) for f, d in completed_tasks
            if not (f == fn.strip() and d == desc.strip())
        ]
        QMessageBox.information(self, "Patch Declined", f"Declined patch for {fn.strip()}")

    def on_queue_feature(self):
        text = self.feature_input.text().strip()
        if text:
            self.feature_list.addItem(text)
            self.feature_input.clear()

    def on_save_features(self):
        # Save to memory/feature_requests.jsonl
        path = os.path.join("memory", "feature_requests.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            for idx in range(self.feature_list.count()):
                req = self.feature_list.item(idx).text()
                f.write(json.dumps({"feature": req}) + "\n")
        QMessageBox.information(self, "Saved", f"Saved {self.feature_list.count()} feature requests.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = GuiMainWindow()
    win.show()
    sys.exit(app.exec_())
