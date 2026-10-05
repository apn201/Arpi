"""The damage report: facts from measurements, model text only if it is a
short report with no advice in it."""
from arpi import report

SCAN = {
    "status": "unique-in-list", "path": "arpi", "frames": 4, "live_count": 1,
    "opencv": {"read": "0350038589900", "verdict": "not in your list"},
    "opencv_detected": True,
    "diagnostics": {"warp_notes": ["break between digit 9 and digit 10, +2.2 modules"],
                    "impossible_runs": 12, "glare_fraction": 0.0, "sharpness": 0.9,
                    "contrast": 0.4},
    "per_digit": [{"position": i + 1, "confidence": 0.99} for i in range(13)],
    "printed_digits": {"text": "03500385851?8"},
}
SCAN["per_digit"][9] = {"position": 10, "confidence": 0.2}
SCAN["per_digit"][10] = {"position": 11, "confidence": 0.3}


def test_facts_come_from_measurements():
    f = report.facts(SCAN)
    assert f[0].startswith("Break between digit 9 and digit 10")
    assert "12 of 95 modules masked or missing." in f
    assert "Bars unreadable at positions 10-11." in f
    assert "12 of 13 printed digits readable." in f
    assert any("wrong code" in x for x in f)
    assert any("only one code in the list fits" in x for x in f)


def test_template_never_contains_the_code():
    assert "0350038589900" not in report.template(SCAN)


class Fake:
    def __init__(self, text):
        self.text, self.calls = text, 0

    def converse(self, **kw):
        self.calls += 1
        assert "do not give advice" in kw["system"][0]["text"].lower()
        return {"output": {"message": {"content": [{"text": self.text}]}}}


def test_model_report_is_used_when_it_is_a_report():
    text = "Torn between digits 9 and 10 with 12 modules missing. Reconstructed from the list."
    out = report.write(SCAN, Fake(text), "model")
    assert out["source"] == "bedrock" and out["report"] == text


def test_advice_is_rejected():
    out = report.write(SCAN, Fake("The label is torn. You should try another angle."), "model")
    assert out["source"] == "template" and "rejected" in out


def test_long_or_numbered_text_is_rejected():
    assert not report.acceptable("word " * 60)
    assert not report.acceptable("Read 0350038585148 after reconstruction.")


def test_no_model_means_template():
    assert report.write(SCAN)["source"] == "template"
