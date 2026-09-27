import sys, time, json
sys.path.insert(0,'scratch')
from ft import call
JOB=sys.argv[1] if len(sys.argv)>1 else "ftjob-27b38b1a5b6d4e52b00e3dcd7375a16b"
API="2025-04-01-preview"
last=None
for _ in range(480):                       # max ~4 h
    s,d = call("GET", f"/openai/fine_tuning/jobs/{JOB}", api=API)
    st=d.get("status")
    if st!=last:
        print(f"[{time.strftime('%H:%M:%S')}] {st} | trained_tokens={d.get('trained_tokens')}", flush=True)
        last=st
    if st in ("succeeded","failed","cancelled"):
        print(json.dumps({k:d.get(k) for k in
             ("status","fine_tuned_model","trained_tokens","created_at","finished_at","error")}, indent=1), flush=True)
        break
    time.sleep(30)
