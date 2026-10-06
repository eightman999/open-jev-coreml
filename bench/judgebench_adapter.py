"""JudgeBench -> typed-decision adapter.

Maps a ScalerLab/JudgeBench pair (question, response_A, response_B, label)
onto the shared {state, questions} interface used by both historical Jev
(api.typesafe.ai systemone) and JevLite (DecisionEngine).

Identical task for every system under test:

    state:     QUESTION / CANDIDATE A / CANDIDATE B  (no labels, no gold)
    questions: choice over {candidate_a, candidate_b, uncertain}

`label` (A>B / B>A) is NEVER included in state or questions.

Pass convention: pass 1 shows (A=original A, B=original B); pass 2 swaps the
displayed order. `canonical_winner` maps a displayed-position verdict back to
the original response identity.
"""
from __future__ import annotations

JB_INSTR = (
    "You are comparing two candidate responses to the question in the state. "
    "Choose the label of the response that is objectively more correct. "
    "Choose uncertain only if the candidates cannot be reliably ranked."
)

JB_CRITERIA = {
    "candidate_a": "candidate A is objectively more correct",
    "candidate_b": "candidate B is objectively more correct",
    "uncertain": "cannot decide reliably from the supplied information",
}

JEV_MAX_STATE_CHARS = 6000          # historical judge_jev.py convention


def jb_questions() -> dict:
    return {"judge": {"type": "choice", "instructions": JB_INSTR,
                      "criteria": dict(JB_CRITERIA)}}


def jb_state(pair: dict, swapped: bool) -> str:
    a = pair["response_B"] if swapped else pair["response_A"]
    b = pair["response_A"] if swapped else pair["response_B"]
    return (f"QUESTION:\n{pair['question']}\n\n"
            f"CANDIDATE A:\n{a}\n\nCANDIDATE B:\n{b}")


def canonical_winner(label: str, swapped: bool) -> str | None:
    """Displayed-position label -> original response identity.

    candidate_a refers to whichever response is displayed in slot A.
    Returns "A"/"B" (original) for decisive labels, "uncertain" for
    abstention, and None for unrecognized output.
    """
    if label == "uncertain":
        return "uncertain"
    if label == "candidate_a":
        return "B" if swapped else "A"
    if label == "candidate_b":
        return "A" if swapped else "B"
    return None


def gold_winner(label: str) -> str:
    """JudgeBench label -> original winner. 'A>B' -> 'A'."""
    if label == "A>B":
        return "A"
    if label == "B>A":
        return "B"
    raise ValueError(f"unknown JudgeBench label {label!r}")


# Terminal-but-not-a-decision categories; never folded into "uncertain".
FAILURE_KINDS = ("ERROR", "TIMEOUT", "MALFORMED", "CRASH")
