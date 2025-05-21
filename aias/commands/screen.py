import os
import pyautogui
from datetime import datetime
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def capture_screenshot(save_path=None):
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	filename = f"screenshot_{timestamp}.png"
	folder = "memory/screenshots"
	os.makedirs(folder, exist_ok=True)

	path = save_path or os.path.join(folder, filename)
	image = pyautogui.screenshot()
	image.save(path)

	print(f"📸 Screenshot saved to {path}")
	return path


def read_screen_text():
	print("🔍 Taking screenshot for OCR...")
	path = capture_screenshot()
	img = Image.open(path)

	try:
		text = pytesseract.image_to_string(img)
		return text.strip()
	except Exception as e:
		print(f"❌ OCR failed: {e}")
		return ""
