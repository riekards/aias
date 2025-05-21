import os
from datetime import datetime
import difflib
import yaml

CONFIG = None

def load_config_if_needed():
	global CONFIG
	if CONFIG is None:
		from utils.config import load_config
		CONFIG = load_config()


def generate_patch_note(old_content, new_content, file_path):
	load_config_if_needed()

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	patch_file = f"memory/patch_notes/{os.path.basename(file_path)}_{timestamp}.patch"

	diff = difflib.unified_diff(
		old_content.splitlines(keepends=True),
		new_content.splitlines(keepends=True),
		fromfile=f"{file_path} (old)",
		tofile=f"{file_path} (new)"
	)

	with open(patch_file, "w", encoding="utf-8") as f:
		f.writelines(diff)

	print(f"📝 Patch note saved: {patch_file}")
	return patch_file


def safe_update_file(file_path, new_content):
	load_config_if_needed()

	# Check extension restrictions
	ext = os.path.splitext(file_path)[1].lower()
	if ext in CONFIG["access"]["restricted_extensions"]:
		print(f"🚫 Write blocked (restricted extension): {file_path}")
		return False

	# Read current content
	old_content = ""
	if os.path.exists(file_path):
		with open(file_path, "r", encoding="utf-8") as f:
			old_content = f.read()

	# If content is identical, skip
	if old_content.strip() == new_content.strip():
		print(f"🔄 No change needed: {file_path}")
		return True

	# Create patch note
	patch_path = generate_patch_note(old_content, new_content, file_path)

	# If patch approval is required
	if CONFIG["modes"].get("patch_approval", True):
		confirm = input(f"❓ Approve changes to {file_path}? (y/n): ").strip().lower()
		if confirm != "y":
			print("❌ Patch rejected by user.")
			return False

	# Apply patch
	with open(file_path, "w", encoding="utf-8") as f:
		f.write(new_content)

	print(f"✅ File updated: {file_path}")
	return True
