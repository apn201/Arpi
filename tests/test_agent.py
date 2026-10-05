"""The agent's policy. No model is called: Bedrock is mocked."""
from arpi import agent


def scan(**over):
    base = {"found": True, "status": "reconstructed", "frames": 1, "live_count": 4,
            "diagnostics": {"glare_fraction": 0.0, "sharpness": 0.9,
                            "left_quality": 0.9, "right_quality": 0.85},
            "per_digit": [{"confidence": 0.99}] * 13,
            "printed_digits": {"text": "6410405060457"}}
    base.update(over)
    return base


def test_read_stops():
    assert agent.decide_rules(scan(status="read"))["action"] == "stop"


def test_glare_asks_for_a_tilt_and_cites_it():
    d = agent.decide_rules(scan(diagnostics={"glare_fraction": 0.08, "sharpness": 0.9}))
    assert d["action"] == "reshoot" and "glare" in d["request"].lower()
    assert d["evidence"] == {"glare_fraction": 0.08}


def test_weak_half_is_named():
    d = agent.decide_rules(scan(diagnostics={"left_quality": 0.2, "right_quality": 0.9,
                                             "sharpness": 0.9}))
    assert "left half" in d["request"]


def test_unread_positions_ask_for_the_printed_digits():
    pd = [{"confidence": 0.99}] * 13
    pd[4] = {"confidence": 0.1}
    d = agent.decide_rules(scan(per_digit=pd, printed_digits={"text": "6410?05060457"}))
    assert d["action"] == "show_digits" and d["evidence"]["unread_positions"] == [5]


def test_gives_up_after_enough_frames():
    assert agent.decide_rules(scan(frames=agent.MAX_FRAMES))["action"] == "stop"


class FakeBedrock:
    def __init__(self, reply=None, fail=False):
        self.reply, self.fail, self.calls = reply, fail, 0

    def converse(self, **kw):
        self.calls += 1
        if self.fail:
            raise RuntimeError("throttled")
        assert "never state or guess the code" in kw["system"][0]["text"].lower()
        return {"output": {"message": {"content": [{"toolUse": {"input": self.reply}}]}}}


def test_bedrock_decision_is_used_and_labelled():
    reply = {"action": "reshoot", "request": "Move closer.", "reason": "grid_fit 0.6",
             "evidence": {"grid_fit": 0.6}}
    d = agent.decide(scan(), FakeBedrock(reply), "model")
    assert d["source"] == "bedrock" and d["request"] == "Move closer."


def test_bedrock_failure_falls_back_to_rules():
    d = agent.decide(scan(), FakeBedrock(fail=True), "model")
    assert d["source"] == "rules" and "failed" in d["note"]


def test_no_model_call_when_the_rules_say_stop():
    fake = FakeBedrock({"action": "reshoot", "request": "x", "reason": "", "evidence": {}})
    agent.decide(scan(status="read"), fake, "model")
    assert fake.calls == 0


def test_budget_spent_means_rules():
    d = agent.decide(scan(), FakeBedrock({}), "model", budget=lambda: False)
    assert d["source"] == "rules" and "budget" in d["note"]
