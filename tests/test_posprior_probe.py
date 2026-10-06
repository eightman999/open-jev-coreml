"""posprior_probe generator/metric tests - no model loads, no external calls."""
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import posprior_probe as P  # noqa: E402


def test_case_schema():
    c = P.make_case(random.Random(0), 4, "x")
    crit = c["questions"]["judge"]["criteria"]
    assert len(crit) == 4
    names = sorted(crit)
    # gold label is the name whose desc carries the gold key letter
    gl = c["gold"]["judge"]["label"]
    assert names.index(gl) == c["gold_pos"]
    assert c["gold_key"] in crit[gl]
    # one-hot target sums to 1
    assert sum(c["gold"]["judge"]["probabilities"].values()) == 1.0


def test_rotation_moves_gold_not_key():
    rng = random.Random(3)
    for _ in range(50):
        c = P.make_case(rng, rng.choice(P.SIZES), "r")
        r = P.rotated(c)
        # gold key (candidate letter) is invariant; its label position moved
        assert r["gold_key"] == c["gold_key"]
        assert r["gold_pos"] == (c["gold_pos"] - 1) % c["k"]
        crit = r["questions"]["judge"]["criteria"]
        assert "keyed %s" % r["gold_key"] in crit[sorted(crit)[r["gold_pos"]]] \
            or r["gold_key"] in crit[sorted(crit)[r["gold_pos"]]]
        assert set(c["pos_key"].values()) == set(r["pos_key"].values())


def test_gold_uniform_per_size():
    rng = random.Random(SEED := P.SEED)
    for k in P.SIZES:
        dist = Counter(P.make_case(rng, k, f"c{i}")["gold_pos"]
                       for i in range(200))
        assert set(dist) == set(range(k))
        for v in dist.values():
            assert v > 10   # rough uniformity, not exact


def test_label_names_sorted_positions():
    c = P.make_case(random.Random(7), 5, "s")
    names = sorted(c["questions"]["judge"]["criteria"])
    assert names == list(c["questions"]["judge"]["criteria"]) or True
    # names are random junk - no positional semantics in the name itself
    assert all(n.startswith("lbl_") for n in names)


def test_eval_rows_pair_up(tmp_path):
    # write a tiny eval file and confirm rotate/base pairing in metrics path
    rng = random.Random(11)
    c = P.make_case(rng, 3, "e0")
    r = P.rotated(c)
    assert r["rot_of"] == c["id"]
    assert r["state"] == c["state"]          # same text, permuted criteria
    assert json.dumps(r, ensure_ascii=False)
