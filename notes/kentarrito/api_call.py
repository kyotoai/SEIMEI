import os
import requests

API_KEY = os.environ["GOOGLE_CSE_API_KEY"]   # set this in your env
CX = os.environ["GOOGLE_CSE_CX"]             # your Search engine ID (cx)

def google_search(query: str, num: int = 5):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": CX,
        "q": query,
        "num": num,          # max 10 per request
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

if __name__ == "__main__":
    data = google_search("Kyoto University", num=5)
    items = data.get("items", [])
    for i, it in enumerate(items, 1):
        print(f"{i}. {it.get('title')}")
        print(f"   {it.get('link')}")
        print(f"   {it.get('snippet')}\n")