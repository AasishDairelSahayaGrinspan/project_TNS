"""Frontend contract tests for loan-approval Streamlit app (port 8502)."""
from pathlib import Path

APP = Path(__file__).resolve().parent / "app.py"


def _src():
    return APP.read_text()


def test_app_file_exists():
    assert APP.exists(), "app.py must exist (resource ownership loan-approval-app/app.py)"


def test_collects_four_applicant_fields():
    src = _src()
    for field in ("income", "credit_score", "loan_amount", "employment_years"):
        assert field in src, f"missing applicant field {field}"


def test_posts_to_backend_predict_8001():
    src = _src()
    assert "http://localhost:8001" in src
    assert "/predict" in src


def test_displays_approval_visually():
    src = _src()
    assert "APPROVED" in src and "DECLINED" in src


def test_landing_first_structure():
    src = _src().lower()
    for section in ("loan approval predictor", "features", "how it works", "how to run"):
        assert section in src, f"missing landing section {section}"


def test_no_css_rgba_strings():
    # Matplotlib tuple colors only; CSS must not use rgba() strings.
    assert "rgba(" not in _src(), "CSS rgba() strings are forbidden"


def test_matplotlib_tuple_colors():
    src = _src()
    assert "matplotlib" in src
    # At least one tuple color definition like (0.62, 0.51, 0.96)
    import re
    assert re.search(r"\(\s*0\.\d+,\s*0\.\d+,\s*0\.\d+", src), "expected tuple color"
