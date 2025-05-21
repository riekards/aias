import requests
from bs4 import BeautifulSoup

def search_google(query):
	print(f"🔍 Searching: {query}")
	url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
	headers = {
		"User-Agent": "Mozilla/5.0"
	}

	try:
		res = requests.get(url, headers=headers)
		soup = BeautifulSoup(res.text, "html.parser")
		results = soup.find_all("a", {"class": "result__a"})

		if not results:
			print("❌ No results found.")
			return

		print("📄 Top Results:")
		for i, link in enumerate(results[:5], start=1):
			print(f"{i}. {link.text.strip()}")
			print(f"   {link['href']}")
	except Exception as e:
		print(f"❌ Search failed: {e}")
