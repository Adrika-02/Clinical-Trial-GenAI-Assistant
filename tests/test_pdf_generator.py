from pathlib import Path

from src.reports.pdf_generator import generate_executive_pdf, _get_demographics


def test_get_demographics_returns_expected_keys():
    demo = _get_demographics()
    assert demo["n_total"] > 0
    assert demo["n_drug_x"] + demo["n_placebo"] == demo["n_total"]
    assert 0 <= demo["completion_rate"] <= 100


def test_generate_executive_pdf_creates_valid_pdf(tmp_path):
    output = tmp_path / "test_report.pdf"
    result_path = generate_executive_pdf(output_path=output, include_llm_insights=False)
    assert result_path.exists()
    assert result_path.stat().st_size > 1000
    with open(result_path, "rb") as f:
        assert f.read(5) == b"%PDF-"
