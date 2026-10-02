#!/usr/bin/env bash
# Three lanes in parallel, one per Azure quota pool; each run deploys and tears down.
cd "$(dirname "$0")/.."
lane() { for m in "$@"; do .venv/bin/python -m benchmark.run --model "$m" --deploy --teardown || echo "FAILED $m"; done; }
lane ces25-sem-20k ces25-sem-50k ces25-sem-100k > logs/bench_lane_sem.log 2>&1 &
lane ces25-fix-20k ces25-fix-50k ces25-fix-100k > logs/bench_lane_fix.log 2>&1 &
lane llama33-70b-base > logs/bench_lane_base.log 2>&1 &
wait
echo "all lanes done"
