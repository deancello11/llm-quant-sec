from src.refusal import is_refusal
from src.judge import parse_guard


def test_refusals_detected():
    for t in ["I'm sorry, but I can't help with that.", "I cannot provide instructions for that.",
              "I apologize, but I won't assist.", "As an AI, I am unable to do this."]:
        assert is_refusal(t), t


def test_compliance_not_flagged():
    assert not is_refusal("Sure! Here is how to bake bread. Step 1: mix flour.")
    assert not is_refusal("Here are the steps. " + "x" * 400 + " I'm sorry for the long text.")


def test_parse_guard():
    assert parse_guard("safe") == (False, [])
    assert parse_guard("unsafe\nS1,S9") == (True, ["S1", "S9"])
    assert parse_guard("") == (False, [])
