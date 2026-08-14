"""Just makes sure the narrated demo script actually runs top to bottom
without crashing, and says something sensible at the end."""

import io
from contextlib import redirect_stdout

from agentlens.scripts.demo import main


def test_demo_runs_and_reports_a_saving():
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    output = buf.getvalue()
    assert "RESULT:" in output
    assert "fewer tool call(s)" in output
