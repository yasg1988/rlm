import asyncio
import shutil
import subprocess
from collections import defaultdict
from typing import Any

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary


class CodexClient(BaseLM):
    """
    LM client that routes completions through the local Codex CLI.

    This backend is intended for personal/local research workflows where Codex is
    already authenticated (for example through a ChatGPT-managed subscription).
    It is not a raw model API: every completion starts a non-interactive Codex
    agent turn and returns the final message from stdout.
    """

    def __init__(
        self,
        model_name: str | None = None,
        codex_bin: str = "codex",
        cwd: str | None = None,
        sandbox: str = "read-only",
        approval_policy: str = "never",
        ephemeral: bool = True,
        skip_git_repo_check: bool = False,
        extra_args: list[str] | None = None,
        timeout: float = 300.0,
        sampling_args: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name or "codex",
            timeout=timeout,
            sampling_args=sampling_args,
            **kwargs,
        )
        self.codex_bin = codex_bin
        self.cwd = cwd
        self.sandbox = sandbox
        self.approval_policy = approval_policy
        self.ephemeral = ephemeral
        self.skip_git_repo_check = skip_git_repo_check
        self.extra_args = list(extra_args or [])
        self.model_call_counts: dict[str, int] = defaultdict(int)

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        prompt_text = self._prompt_to_text(prompt)
        command = self._build_command(model=model)

        completed = subprocess.run(
            command,
            input=prompt_text.encode("utf-8"),
            capture_output=True,
            cwd=self.cwd,
            timeout=self.timeout,
            check=False,
        )

        active_model = model or self.model_name
        self.model_call_counts[active_model] += 1
        stdout = completed.stdout.decode("utf-8", errors="replace").strip()
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        if completed.returncode != 0:
            details = stderr or stdout
            raise RuntimeError(
                f"Codex CLI failed with exit code {completed.returncode}"
                + (f": {details}" if details else "")
            )

        return stdout

    async def acompletion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        return await asyncio.to_thread(self.completion, prompt, model)

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(
            model_usage_summaries={
                model: ModelUsageSummary(
                    total_calls=calls,
                    total_input_tokens=0,
                    total_output_tokens=0,
                    total_cost=None,
                )
                for model, calls in self.model_call_counts.items()
            }
        )

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost=None,
        )

    def _build_command(self, model: str | None = None) -> list[str]:
        command = [self._resolve_codex_bin()]
        if self.approval_policy:
            command.extend(["--ask-for-approval", self.approval_policy])
        command.append("exec")
        if self.ephemeral:
            command.append("--ephemeral")
        if self.sandbox:
            command.extend(["--sandbox", self.sandbox])
        if self.skip_git_repo_check:
            command.append("--skip-git-repo-check")
        if model:
            command.extend(["--model", model])
        elif self.model_name and self.model_name != "codex":
            command.extend(["--model", self.model_name])
        command.extend(self.extra_args)
        command.append("-")
        return command

    def _resolve_codex_bin(self) -> str:
        return shutil.which(self.codex_bin) or self.codex_bin

    @staticmethod
    def _prompt_to_text(prompt: str | list[dict[str, Any]]) -> str:
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            parts = []
            for item in prompt:
                role = str(item.get("role", "message"))
                content = item.get("content", "")
                parts.append(f"{role.upper()}:\n{content}")
            return "\n\n".join(parts)
        raise ValueError(f"Invalid prompt type: {type(prompt)}")
