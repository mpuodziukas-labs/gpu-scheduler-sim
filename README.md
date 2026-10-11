# gpu-scheduler-sim

Shows how queue policy choice (FIFO, EASY backfill, gang-aware bin-packing) changes GPU utilization, wait time and fragmentation on a shared accelerator cluster, using a seeded simulation.

This is a SIMULATION. It needs no GPU, runs on the Python 3 standard library, and costs nothing to run.

## Run it in 60 seconds

```
python3 evaluate.py
python3 -m pip install pytest && python3 -m pytest -q
```

`evaluate.py` finishes in about a second and exits nonzero if any scheduling invariant fails. Output:

```
SIMULATION (synthetic workload, not a production scheduler): seed=7, 500 jobs, 8 nodes x 8 GPUs

| policy | utilization | mean wait (min) | p95 wait (min) | makespan (min) | fragmentation | invariants |
|---|---|---|---|---|---|---|
| fifo | 77.9% | 94.9 | 252 | 1503 | 38.3% | PASS |
| easy_backfill | 82.2% | 41.4 | 153 | 1424 | 33.3% | PASS |
| gang_binpack | 83.0% | 33.0 | 189 | 1410 | 37.7% | PASS |

output sha256: b56e27b0507a64ab
```

Columns: utilization = GPU-minutes of work / (total GPUs x makespan). Wait = start minus arrival, p95 by rank. Fragmentation = time-weighted share of idle GPUs that are not available as one block on a single node, measured only while jobs are waiting. The sha256 line is the hash of the table text, so a rerun with the same seed must reproduce it exactly.

## What is modelled

- 8 nodes x 8 GPUs. 500 jobs from a seeded generator (arrival, GPUs requested from 1 to 32, duration, priority 0 to 2).
- Jobs of up to 8 GPUs run on one node. Jobs of 16 and 32 GPUs are gangs: all of their nodes are allocated together on whole nodes, or the job does not start.
- `fifo`: strict arrival order, the head job blocks the queue.
- `easy_backfill`: FIFO plus EASY backfill. The head job gets a reserved start time; a later job may jump ahead only if it is placeable now and ends before that reservation. Durations are treated as exact.
- `gang_binpack`: priority first, then largest job first so gangs claim whole nodes early; sub-node jobs are best-fit onto the fullest node that still fits.

## Invariants (checked by an independent replay of the schedule log)

1. No node is ever over-allocated.
2. Every job starts with its full allocation (all-or-nothing), and gangs sit on whole nodes.
3. Backfill never starts the reserved head job later than its reservation.
4. Same seed gives an identical output hash; a different seed gives a different one.

Three planted buggy policies (over-allocation, partial gang start, backfill that ignores the reservation) live in `sim.py`; the tests assert the checker reports each of them.

## Honesty Statement

Results here are exactly what `python3 evaluate.py` printed for seed 7 on a synthetic workload. They say how these three policies behave on that generated load, not how any real cluster or vendor scheduler behaves. Other seeds, loads and job mixes can change the ranking.

## Limitations

- Synthetic workload only; no production traces.
- A toy model, not a production scheduler: no preemption, no job migration, no topology or interconnect awareness, no failures, no fair-share quotas, no multi-tenant isolation.
- Backfill uses exact job durations; real users misestimate runtimes.
- `gang_binpack` has no starvation protection for low-priority or small jobs, so its p95 wait can be worse than backfill's, as the table shows.
- Single seed in the README table; the tests check invariants, not statistical significance.
