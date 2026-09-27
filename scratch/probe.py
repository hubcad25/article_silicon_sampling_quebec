import os, urllib.request, urllib.error
from dotenv import load_dotenv
load_dotenv(".env")
ep = os.environ.get("FOUNDRY_CHAT_ENDPOINT","").rstrip("/")
key = os.environ.get("FOUNDRY_CHAT_KEY","")
print("endpoint:", ep or "(absent)", "| key present:", bool(key))
def get(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"api-key":key}), timeout=30) as r:
            return r.status, r.read()[:200].decode(errors="replace")
    except urllib.error.HTTPError as e: return e.code, e.read()[:200].decode(errors="replace")
    except Exception as e: return "ERR", str(e)[:150]
base = ep.split("/openai")[0].split("/models")[0]
print("base:", base)
for api in ["2024-10-21","2025-04-01-preview","2025-08-01-preview"]:
    print(f"  files {api:20} ->", get(f"{base}/openai/files?api-version={api}"))
