import requests

def fetch_url(url):
	if not url.startswith("http"):
		url = "https://" + url

	try:
		print(f"🌐 Fetching: {url}")
		response = requests.get(url, timeout=10)
		print(f"✅ Status: {response.status_code}")
		print("📄 Response preview:\n", response.text[:1000])  # limit output
		return response.text
	except Exception as e:
		print(f"❌ Failed to fetch: {e}")
		return ""
