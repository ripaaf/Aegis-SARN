from __future__ import annotations

import json

from aegis_sarn.cli import main


def test_cli_fake_backend_emits_json_trace(capsys: object) -> None:
    exit_code = main(
        ['run', '--backend', 'fake', '--prompt', 'hello', '--max-new-tokens', '3']
    )
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload['status'] == 'completed'
    assert payload['trace']
    assert payload['backend'] == 'fake'
