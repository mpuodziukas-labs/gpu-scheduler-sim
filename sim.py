"""Discrete-event GPU cluster scheduler SIMULATION (synthetic workload, stdlib only)."""
import heapq, random
from collections import namedtuple

Job = namedtuple("Job", "id arrival gpus duration priority")
Record = namedtuple("Record", "job start end alloc")


class Result:
    def __init__(self, records, reservations, backfilled, frag, nodes, g):
        self.records, self.reservations, self.backfilled = records, reservations, backfilled
        self.frag, self.nodes, self.g = frag, nodes, g


def make_workload(seed, n=500):
    """Whole-minute arrivals/durations. Jobs above one node (16, 32 GPUs) are gangs on whole nodes."""
    rng = random.Random(seed)
    sizes, weights = [1, 2, 4, 8, 16, 32], [30, 20, 20, 15, 10, 5]
    t, jobs = 0, []
    for i in range(n):
        t += int(rng.expovariate(1 / 2.8))
        jobs.append(Job(i, t, rng.choices(sizes, weights)[0],
                        max(1, int(rng.lognormvariate(3.0, 0.8))), rng.choice([0, 1, 2])))
    return jobs


def place(free, job, g, best_fit=False):
    """Allocation {node: gpus} for the whole job, or None. Never splits a sub-node job."""
    if job.gpus <= g:
        cands = [n for n, f in enumerate(free) if f >= job.gpus]
        if not cands:
            return None
        return {(min(cands, key=lambda n: (free[n], n)) if best_fit else cands[0]): job.gpus}
    empty = [n for n, f in enumerate(free) if f == g]
    k = job.gpus // g
    return {n: g for n in empty[:k]} if len(empty) >= k else None


class Ctx:
    def __init__(self, t, queue, free, running, g, state):
        self.t, self.queue, self.free, self.running, self.g, self.state = t, queue, free, running, g, state
        self.started = []

    def start(self, job, alloc):
        for n, k in alloc.items():
            self.free[n] -= k
        self.queue.remove(job)
        heapq.heappush(self.running, (self.t + job.duration, job.id, alloc))
        self.started.append((job, alloc))


def shadow_time(ctx, head):
    """Earliest time the head job could start if running jobs end when they are due to."""
    free = list(ctx.free)
    for end, _, alloc in sorted(ctx.running):
        for n, k in alloc.items():
            free[n] += k
        if place(free, head, ctx.g):
            return end


def fifo(ctx):
    while ctx.queue:
        a = place(ctx.free, ctx.queue[0], ctx.g)
        if not a:
            return
        ctx.start(ctx.queue[0], a)


def _backfill(ctx, respect_shadow):
    fifo(ctx)
    if not ctx.queue:
        return
    head = ctx.queue[0]
    sh = shadow_time(ctx, head)
    ctx.state.setdefault("reservations", {}).setdefault(head.id, sh)
    for j in list(ctx.queue[1:]):
        if respect_shadow and ctx.t + j.duration > sh:
            continue
        a = place(ctx.free, j, ctx.g)
        if a:
            ctx.start(j, a)
            ctx.state["backfilled"] = ctx.state.get("backfilled", 0) + 1


def easy_backfill(ctx):
    _backfill(ctx, True)


def gang_binpack(ctx):
    """Priority, then largest first (gangs claim whole nodes early); small jobs best-fit onto used nodes."""
    for j in sorted(ctx.queue, key=lambda j: (-j.priority, -j.gpus, j.arrival, j.id)):
        a = place(ctx.free, j, ctx.g, best_fit=True)
        if a:
            ctx.start(j, a)


POLICIES = {"fifo": fifo, "easy_backfill": easy_backfill, "gang_binpack": gang_binpack}


# Planted bugs: the invariant checker must catch each of these.
def bug_overallocate(ctx):
    for j in list(ctx.queue):
        if j.gpus <= ctx.g:
            ctx.start(j, {0: j.gpus})  # ignores free capacity
        elif (a := place(ctx.free, j, ctx.g)):
            ctx.start(j, a)


def bug_partial_gang(ctx):
    for j in list(ctx.queue):
        a = place(ctx.free, j, ctx.g)
        if a and j.gpus > ctx.g:
            a = dict(list(a.items())[:-1])  # starts with one node short
        if a:
            ctx.start(j, a)


def bug_greedy_backfill(ctx):
    _backfill(ctx, False)  # computes the reservation, then ignores it


BUGGY = {"BUG_overallocate": bug_overallocate, "BUG_partial_gang": bug_partial_gang,
         "BUG_greedy_backfill": bug_greedy_backfill}


def simulate(jobs, policy, nodes, g):
    free, running, queue, records, state = [g] * nodes, [], [], [], {}
    pending, i, t, last, frag = sorted(jobs, key=lambda j: (j.arrival, j.id)), 0, 0, 0, [0.0, 0.0]
    while i < len(pending) or queue or running:
        nxt = [x for x in (pending[i].arrival if i < len(pending) else None,
                           running[0][0] if running else None) if x is not None]
        t = min(nxt)
        if queue and sum(free) > 0:  # fragmentation: share of idle GPUs not usable as one block, while jobs wait
            frag[0] += (t - last) * (1 - max(free) / sum(free)) if sum(free) else 0
            frag[1] += t - last
        last = t
        while running and running[0][0] == t:
            _, _, alloc = heapq.heappop(running)
            for n, k in alloc.items():
                free[n] += k
        while i < len(pending) and pending[i].arrival == t:
            queue.append(pending[i]); i += 1
        ctx = Ctx(t, queue, free, running, g, state)
        policy(ctx)
        for job, alloc in ctx.started:
            records.append(Record(job, t, t + job.duration, alloc))
    return Result(records, state.get("reservations", {}), state.get("backfilled", 0),
                  frag[0] / frag[1] if frag[1] else 0.0, nodes, g)


def check_invariants(res, nodes, g):
    """Independent replay of the schedule log. Returns a list of violations (empty = clean)."""
    bad, ev = [], []
    for r in res.records:
        j = r.job
        if sum(r.alloc.values()) != j.gpus:
            bad.append(f"job {j.id}: all-or-nothing violated ({sum(r.alloc.values())}/{j.gpus} GPUs at start)")
        if j.gpus > g and any(k != g for k in r.alloc.values()):
            bad.append(f"job {j.id}: gang not on whole nodes")
        if r.start < j.arrival or r.end != r.start + j.duration:
            bad.append(f"job {j.id}: bad timing")
        for n, k in r.alloc.items():
            ev += [(r.start, 1, n, k, j.id), (r.end, 0, n, -k, j.id)]
    use = [0] * nodes
    for _, _, n, d, jid in sorted(ev):
        use[n] += d
        if use[n] > g:
            bad.append(f"node {n} over-allocated ({use[n]}/{g}) by job {jid}")
    start = {r.job.id: r.start for r in res.records}
    for jid, reserved in res.reservations.items():
        if start[jid] > reserved:
            bad.append(f"job {jid}: started {start[jid]} after its reservation {reserved}")
    if len({r.job.id for r in res.records}) != len(res.records):
        bad.append("a job ran twice")
    return bad[:5]
