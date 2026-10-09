from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_launcher_executes() -> None:
    path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    app = AppTest.from_file(path).run(timeout=10)
    assert not app.exception
