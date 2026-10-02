#!/usr/bin/env bash
# Re-run the models whose deployment PUT hit a 409 at launch, once their quota pool is free.
cd "$(dirname "$0")/.."
after() { until [ -f "benchmark/runs/$1/run_manifest.json" ]; do sleep 60; done; }
( after ces25-sem-100k; .venv/bin/python -m benchmark.run --model ces25-sem-20k --deploy --teardown ) > logs/bench_catchup_sem.log 2>&1 &
( after ces25-fix-100k; for m in ces25-fix-20k ces25-fix-50k; do .venv/bin/python -m benchmark.run --model $m --deploy --teardown; done ) > logs/bench_catchup_fix.log 2>&1 &
wait
