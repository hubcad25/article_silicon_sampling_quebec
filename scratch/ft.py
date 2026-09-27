import os, json, urllib.request, urllib.error, uuid, sys
from dotenv import load_dotenv
load_dotenv(".env")
BASE = os.environ["FOUNDRY_CHAT_ENDPOINT"].rstrip("/").split("/openai")[0]
KEY  = os.environ["FOUNDRY_CHAT_KEY"]
API  = "2024-10-21"

def call(method, path, body=None, raw=None, ctype=None, api=API):
    url = f"{BASE}{path}{'&' if '?' in path else '?'}api-version={api}"
    hdr = {"api-key": KEY}
    data = None
    if body is not None:
        data = json.dumps(body).encode(); hdr["Content-Type"] = "application/json"
    if raw is not None:
        data = raw; hdr["Content-Type"] = ctype
    req = urllib.request.Request(url, data=data, headers=hdr, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try: return e.code, json.loads(e.read())
        except Exception: return e.code, {"raw": "unparsed"}

def upload(path, purpose="fine-tune"):
    b = uuid.uuid4().hex
    fn = os.path.basename(path)
    parts = []
    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\n{purpose}\r\n'.encode())
    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{fn}"\r\nContent-Type: application/jsonl\r\n\r\n'.encode())
    parts.append(open(path,"rb").read()); parts.append(f'\r\n--{b}--\r\n'.encode())
    return call("POST", "/openai/files", raw=b"".join(parts), ctype=f"multipart/form-data; boundary={b}")

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "jobs":
        s, d = call("GET", "/openai/fine_tuning/jobs")
        print(s, "| jobs:", len(d.get("data", [])))
        for j in d.get("data", [])[:8]:
            print(f"  {j.get('id')} | {j.get('model')} -> {j.get('fine_tuned_model')} | {j.get('status')} | err={ (j.get('error') or {}).get('message','')[:70] }")
    elif cmd == "upload":
        for p in sys.argv[2:]:
            s, d = upload(p); print(s, p, "->", d.get("id"), d.get("status"), d.get("error",""))

def status_files(ids):
    for i in ids:
        s,d = call("GET", f"/openai/files/{i}")
        print(f"  {i} -> {d.get('status')} {d.get('status_details') or ''} bytes={d.get('bytes')}")
