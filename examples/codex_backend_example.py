"""
Run RLM with the local Codex CLI as the backend.

This uses the existing Codex CLI authentication, including ChatGPT-managed
Codex access. It does not require an OpenAI API key, but it does require that
`codex exec` works on this machine.

Run:
    python -m examples.codex_backend_example
"""

from rlm import RLM

rlm = RLM(
    backend="codex",
    backend_kwargs={
        "sandbox": "read-only",
        "approval_policy": "never",
        "timeout": 300,
    },
    environment="local",
    max_depth=1,
    max_iterations=5,
    max_timeout=600,
    verbose=True,
)

result = rlm.completion(
    "Answer in one short paragraph: what is the difference between Codex CLI "
    "and a raw OpenAI API completion?"
)

print(result.response)
