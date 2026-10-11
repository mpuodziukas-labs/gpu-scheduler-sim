import hashlib, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import pytest
import sim, evaluate

NODES, G = 8, 8
JOBS = sim.make_workload(seed=7, n=500)
GOOD = ["fifo", "easy_backfill", "gang_binpack"]


def run(name, jobs=JOBS, policies=sim.POLICIES):
    return sim.simulate(jobs, policies[name], NODES, G)


def test_workload_shape_and_seed():
    assert len(JOBS) == 500
    assert JOBS == sim.make_workload(seed=7, n=500)
    assert JOBS != sim.make_workload(seed=8, n=500)
    assert any(j.gpus > G for j in JOBS) and any(j.gpus == 1 for j in JOBS)


@pytest.mark.parametrize("name", GOOD)
def test_good_policies_hold_all_invariants(name):
    assert sim.check_invariants(run(name), NODES, G) == []
    assert len(run(name).records) == 500


@pytest.mark.parametrize("name", GOOD)
def test_never_overallocates_a_node(name):
    res = run(name)
    events = []
    for r in res.records:
        for n, k in r.alloc.items():
            events += [(r.start, 1, n, k), (r.end, 0, n, -k)]
    use = [0] * NODES
    for _, _, n, d in sorted(events):
        use[n] += d
        assert 0 <= use[n] <= G


@pytest.mark.parametrize("name", GOOD)
def test_gang_all_or_nothing(name):
    for r in run(name).records:
        assert sum(r.alloc.values()) == r.job.gpus
        if r.job.gpus > G:
            assert all(k == G for k in r.alloc.values())
        else:
            assert len(r.alloc) == 1


def test_backfill_never_delays_head_reservation():
    res = run("easy_backfill")
    assert res.reservations, "no reservation was ever made, test is vacuous"
    by_id = {r.job.id: r for r in res.records}
    for jid, reserved in res.reservations.items():
        assert by_id[jid].start <= reserved


def test_backfill_actually_backfills():
    res = run("easy_backfill")
    assert res.backfilled > 0


def test_determinism_same_seed_same_hash():
    def h(seed):
        return hashlib.sha256(evaluate.report(seed).encode()).hexdigest()
    assert h(7) == h(7)
    assert h(7) != h(8)


def test_planted_overallocation_is_caught():
    res = run("BUG_overallocate", policies=sim.BUGGY)
    bad = sim.check_invariants(res, NODES, G)
    assert any("over-allocated" in b for b in bad)


def test_planted_partial_gang_is_caught():
    res = run("BUG_partial_gang", policies=sim.BUGGY)
    assert any("all-or-nothing" in b for b in sim.check_invariants(res, NODES, G))


def test_planted_greedy_backfill_is_caught():
    res = run("BUG_greedy_backfill", policies=sim.BUGGY)
    assert any("reservation" in b for b in sim.check_invariants(res, NODES, G))


def test_evaluate_exits_nonzero_if_invariant_fails(monkeypatch):
    monkeypatch.setitem(sim.POLICIES, "fifo", sim.BUGGY["BUG_overallocate"])
    assert evaluate.main([]) != 0


def test_readme_matches_output():
    readme = (pathlib.Path(__file__).parent.parent / "README.md").read_text()
    assert evaluate.report(evaluate.SEED).strip() in readme, "README output drifted from evaluate.py"
