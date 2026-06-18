"""Run an RLM query using the local Codex CLI as the model backend."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rlm import RLM
from rlm.clients.codex import CodexClient
from rlm.logger import RLMLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="Prompt text. Reads stdin when omitted.")
    parser.add_argument("--prompt-file", type=Path, help="Read prompt text from a UTF-8 file.")
    parser.add_argument("--cwd", type=str, default=None, help="Working directory for codex exec.")
    parser.add_argument("--model", type=str, default=None, help="Optional Codex model override.")
    parser.add_argument("--sandbox", type=str, default="read-only", help="Codex sandbox mode.")
    parser.add_argument(
        "--timeout", type=float, default=300.0, help="Per Codex call timeout seconds."
    )
    parser.add_argument("--max-depth", type=int, default=1, help="Maximum RLM recursion depth.")
    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum RLM iterations.")
    parser.add_argument(
        "--max-timeout", type=float, default=900.0, help="Whole RLM timeout seconds."
    )
    parser.add_argument(
        "--log-dir", type=Path, default=None, help="Optional RLM trajectory log dir."
    )
    parser.add_argument("--verbose", action="store_true", help="Print RLM execution details.")
    parser.add_argument(
        "--skip-git-repo-check",
        action="store_true",
        help="Pass --skip-git-repo-check to codex exec.",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Bypass recursive RLM and send the prompt directly to Codex CLI.",
    )
    parser.add_argument(
        "--auto-direct-fallback",
        action="store_true",
        help="Retry with direct Codex when RLM cannot access its context.",
    )
    return parser.parse_args()


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return args.prompt_file.read_text(encoding="utf-8")
    if args.prompt:
        return args.prompt
    return sys.stdin.read()


def main() -> int:
    args = parse_args()
    prompt = read_prompt(args).strip()
    if not prompt:
        print("Prompt is empty", file=sys.stderr)
        return 2

    logger = RLMLogger(log_dir=str(args.log_dir)) if args.log_dir else None
    backend_kwargs = {
        "sandbox": args.sandbox,
        "approval_policy": "never",
        "timeout": args.timeout,
        "cwd": args.cwd,
        "skip_git_repo_check": args.skip_git_repo_check,
    }
    if args.model:
        backend_kwargs["model_name"] = args.model

    if args.direct:
        print(run_direct_codex(prompt, backend_kwargs))
        return 0

    rlm = RLM(
        backend="codex",
        backend_kwargs=backend_kwargs,
        environment="local",
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
        max_timeout=args.max_timeout,
        logger=logger,
        verbose=args.verbose,
    )
    result = rlm.completion(prompt)
    if args.auto_direct_fallback and looks_like_missing_context(result.response):
        print(run_direct_codex(prompt, backend_kwargs))
        return 0
    print(result.response)
    return 0


def looks_like_missing_context(response: str) -> bool:
    text = response.lower()
    markers = [
        "context was never available",
        "context variable",
        "could not inspect",
        "no repl tool",
        "necessary `context` was never available",
    ]
    return any(marker in text for marker in markers)


def run_direct_codex(prompt: str, backend_kwargs: dict) -> str:
    client = CodexClient(**backend_kwargs)
    direct_prompt = (
        "You are doing read-only research. Do not modify files. "
        "Answer directly from the prompt content and any read-only repository inspection "
        "allowed by the sandbox. If context is included in the prompt, treat it as the source of truth.\n\n"
        + prompt
    )
    return client.completion(direct_prompt)


if __name__ == "__main__":
    raise SystemExit(main())
