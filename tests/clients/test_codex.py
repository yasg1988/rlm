from unittest.mock import patch

import pytest

from rlm.clients import get_client
from rlm.clients.codex import CodexClient


def test_codex_client_invokes_codex_exec_with_stdin():
    client = CodexClient(codex_bin="codex-test", cwd=".", timeout=12.0)

    with patch("rlm.clients.codex.subprocess.run") as run:
        run.return_value.returncode = 0
        run.return_value.stdout = b"final answer\n"
        run.return_value.stderr = b""

        assert client.completion("hello") == "final answer"

    command = run.call_args.args[0]
    assert command == [
        "codex-test",
        "--ask-for-approval",
        "never",
        "exec",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "-",
    ]
    assert run.call_args.kwargs["input"] == b"hello"
    assert run.call_args.kwargs["cwd"] == "."
    assert run.call_args.kwargs["timeout"] == 12.0


def test_codex_client_supports_model_and_extra_args():
    client = CodexClient(
        model_name="gpt-test",
        sandbox="workspace-write",
        extra_args=["--json"],
    )

    with patch("rlm.clients.codex.subprocess.run") as run:
        run.return_value.returncode = 0
        run.return_value.stdout = b"ok"
        run.return_value.stderr = b""

        client.completion("hello")

    command = run.call_args.args[0]
    assert "--model" in command
    assert "gpt-test" in command
    assert "--json" in command
    assert command[-1] == "-"


def test_codex_client_raises_on_nonzero_exit():
    client = CodexClient()

    with patch("rlm.clients.codex.subprocess.run") as run:
        run.return_value.returncode = 2
        run.return_value.stdout = b""
        run.return_value.stderr = b"boom"

        with pytest.raises(RuntimeError, match="boom"):
            client.completion("hello")


def test_get_client_supports_codex_backend():
    client = get_client("codex", {"codex_bin": "codex-test"})
    assert isinstance(client, CodexClient)
