"""td_data.encode serialization tests - tokenizer-level, no model loads."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import td_data  # noqa: E402


class _ToyTok:
    """Whitespace tokenizer with fixed ids - enough for layout assertions."""
    cls_token_id, sep_token_id, pad_token_id = 1, 2, 0
    _v = {}

    @classmethod
    def encode(cls, text, add_special_tokens=False):
        ids = []
        for w in text.split():
            if w not in cls._v:
                cls._v[w] = len(cls._v) + 100
            ids.append(cls._v[w])
        return ids


TOK = _ToyTok()
QID, LID = 7, 8

ROW = {"id": "x", "state": "some state text",
       "questions": {"q1": {"type": "choice",
                            "instructions": "pick one",
                            "criteria": {"b": "desc b", "a": "desc a",
                                         "c": "desc c"}},
                     "q2": {"type": "score",
                            "instructions": "rate it",
                            "criteria": ["bad", "ok", "good"]}},
       "gold": {"q1": {"label": "c",
                       "probabilities": {"a": 0.0, "b": 0.0, "c": 1.0}},
                "q2": {"label": "2",
                       "probabilities": {"0": 0.0, "1": 0.0, "2": 1.0}}}}


def _markers_at(enc, ids):
    return [i for i, t in enumerate(ids) if t == LID]


def test_sym_lead_slot_not_gathered():
    enc = td_data.encode(ROW, TOK, 512, QID, LID)
    lids = _markers_at(enc, enc["input_ids"])
    # 3 choice labels + 3 score labels = 6 gathered; +2 lead slots = 8 markers
    assert len(lids) == 8
    assert len(enc["label_pos"]) == 6
    assert set(enc["label_pos"]).issubset(set(lids))
    lead_txt = TOK.encode(td_data.LEAD_SLOT_TEXT)
    # each question's first gathered marker is preceded by
    # `<<l>> LEAD_SLOT_TEXT` - a label-shaped left neighbour, not instructions
    for p in (enc["label_pos"][0], enc["label_pos"][3]):
        n = len(lead_txt)
        assert enc["input_ids"][p - n - 1] == LID
        assert enc["input_ids"][p - n:p] == lead_txt


def test_flat_layout_unchanged():
    enc = td_data.encode(ROW, TOK, 512, QID, LID, slots="none")
    lids = _markers_at(enc, enc["input_ids"])
    assert len(lids) == len(enc["label_pos"]) == 6
    # first label marker is preceded by the instruction's last token
    p0 = enc["label_pos"][0]
    assert enc["input_ids"][p0 - 1] == TOK.encode("one")[-1]


def test_every_gathered_marker_is_lid():
    for ser in ("lead", "none", "pair"):
        enc = td_data.encode(ROW, TOK, 512, QID, LID, slots=ser)
        for p in enc["label_pos"]:
            assert enc["input_ids"][p] == LID


def test_pair_slots_give_identical_predecessors():
    enc = td_data.encode(ROW, TOK, 512, QID, LID, slots="pair")
    lead = TOK.encode(td_data.LEAD_SLOT_TEXT)
    n = len(lead)
    # every gathered marker sees the identical `<<l>> SLOT_TEXT` predecessor
    for p in enc["label_pos"]:
        assert enc["input_ids"][p - n - 1] == LID
        assert enc["input_ids"][p - n:p] == lead
    lids = _markers_at(enc, enc["input_ids"])
    assert len(lids) == 14          # 6 labels x (dummy + real) + 2 trailing


def test_permutation_moves_labels_not_gold():
    enc = td_data.encode(ROW, TOK, 512, QID, LID,
                         rng=random.Random(4))
    q1_labels = enc["labels"][0]
    assert sorted(q1_labels) == ["a", "b", "c"]
    # gold target still sits under label "c" wherever it was emitted
    gi = q1_labels.index("c")
    assert enc["target"][gi] == 1.0
    # labels list order matches emission order: marker positions ascend
    assert enc["label_pos"] == sorted(enc["label_pos"])


def test_score_never_permuted():
    for seed in range(20):
        enc = td_data.encode(ROW, TOK, 512, QID, LID,
                             rng=random.Random(seed))
        assert enc["labels"][1] == ["0", "1", "2"]


def test_resample_changes_order_per_epoch():
    ds = td_data.TypedDecisions([ROW], TOK, 512, QID, LID, permute=True)
    seen = set()
    for ep in range(30):
        ds.resample(1000 + ep)
        seen.add(tuple(ds[0]["labels"][0]))
    assert len(seen) > 1


def test_no_permutation_by_default():
    ds = td_data.TypedDecisions([ROW], TOK, 512, QID, LID)
    ds.resample(1)
    ds.resample(2)
    assert ds[0]["labels"][0] == ["a", "b", "c"]


def test_lead_slot_text_constant():
    # the dummy block must be label-shaped but never a real criterion name
    assert ": " in td_data.LEAD_SLOT_TEXT
    assert td_data.LEAD_SLOT_TEXT.split(":")[0] not in ROW["questions"]["q1"][
        "criteria"]
