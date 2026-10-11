# gpu-scheduler-sim

Shows how queue policy choice (FIFO, EASY backfill, gang-aware bin-packing) changes GPU utilization, wait time and fragmentation on a shared accelerator cluster, using a seeded simulation.

## Run it in 60 seconds

```
python3 -m pip install pytest
python3 evaluate.py
python3 -m pytest -q
```

## Results

| bad caught | good passed |
|---|---|
| 1/1 | 1/1 |

## Honesty Statement

Synthetic data. Not production, not client data.

## Limitations

Small synthetic corpus. Replace before drawing conclusions.
