import os
import subprocess
import platform

def open_vscode():
	print("🧠 Opening VS Code...")
	try:
		subprocess.Popen("code", shell=True)
	except Exception as e:
		print(f"❌ Could not open VS Code: {e}")


def delete_temp_files():
	temp_dir = os.environ.get("TEMP") or "/tmp"
	print(f"🧹 Cleaning temp folder: {temp_dir}")

	for filename in os.listdir(temp_dir):
		file_path = os.path.join(temp_dir, filename)
		try:
			if os.path.isfile(file_path):
				os.remove(file_path)
			elif os.path.isdir(file_path):
				os.rmdir(file_path)
		except Exception as e:
			print(f"⚠️ Failed to delete {file_path}: {e}")

	print("✅ Temp folder cleaned.")


def shutdown_computer():
	os_name = platform.system().lower()
	if os_name == "windows":
		subprocess.run("shutdown /s /t 5", shell=True)
	elif os_name == "linux" or os_name == "darwin":
		subprocess.run("sudo shutdown now", shell=True)
	else:
		print("⚠️ Unsupported OS for shutdown.")
