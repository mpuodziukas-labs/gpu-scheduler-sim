"""One command: python3 evaluate.py  (prints the policy comparison; exit 1 if any invariant fails)."""
import hashlib, sys
import sim

SEED, JOBS, NODES, G = 7, 500, 8, 8


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def report(seed):
    jobs = sim.make_workload(seed, JOBS)
    rows = ["| policy | utilization | mean wait (min) | p95 wait (min) | makespan (min) | fragmentation | invariants |",
            "|---|---|---|---|---|---|---|"]
    for name, pol in sim.POLICIES.items():
        res = sim.simulate(jobs, pol, NODES, G)
        waits = [r.start - r.job.arrival for r in res.records]
        work = sum(r.job.gpus * r.job.duration for r in res.records)
        span = max(r.end for r in res.records)
        bad = sim.check_invariants(res, NODES, G)
        rows.append(f"| {name} | {work / (NODES * G * span):.1%} | {sum(waits) / len(waits):.1f} | {pct(waits, .95)} "
                    f"| {span} | {res.frag:.1%} | {'PASS' if not bad else 'FAIL ' + bad[0]} |")
    head = (f"SIMULATION (synthetic workload, not a production scheduler): seed={seed}, {JOBS} jobs, "
            f"{NODES} nodes x {G} GPUs\n\n")
    out = head + "\n".join(rows)
    return out + f"\n\noutput sha256: {hashlib.sha256(out.encode()).hexdigest()[:16]}\n"


def main(argv):
    out = report(SEED)
    print(out, end="")
    return 1 if "FAIL" in out.split("\n\n", 1)[1] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
