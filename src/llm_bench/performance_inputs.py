"""Prepare exact-size fresh-context and growing-session inputs with the submitted tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def token_count(tokenizer: Any, messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> int:
    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        **kwargs,
    )
    return len(tokens)


def template_count(tokenizer: Any, messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> int:
    """Count prompt tokens, parsing tool-call arguments as the chat template expects."""
    work = json.loads(json.dumps(messages))
    for message in work:
        for call in message.get("tool_calls", []):
            arguments = call.get("function", {}).get("arguments")
            if isinstance(arguments, str):
                try:
                    call["function"]["arguments"] = json.loads(arguments)
                except ValueError:
                    pass
    return token_count(tokenizer, work, kwargs)


def sized_messages(
    tokenizer: Any,
    input_tokens: int,
    nonce: str,
    template_kwargs: dict[str, Any],
) -> tuple[list[dict[str, Any]], int, int]:
    messages = [
        {
            "role": "system",
            "content": (
                f"real-bench fresh-context request {nonce}. "
                "This prefix is unique to discourage prefix reuse."
            ),
        },
        {
            "role": "user",
            "content": "Analyze the independent log and explain the important patterns.",
        },
    ]
    low, high = 0, input_tokens
    actual = token_count(tokenizer, messages, template_kwargs)
    while low <= high:
        middle = (low + high) // 2
        messages[-1]["content"] = (
            "Analyze the independent log and explain the important patterns." + " entry" * middle
        )
        actual = token_count(tokenizer, messages, template_kwargs)
        if actual == input_tokens:
            break
        if actual < input_tokens:
            low = middle + 1
        else:
            high = middle - 1
    else:
        raise ValueError(
            f"Cannot size fresh-context input exactly to {input_tokens} tokens "
            "with the submitted tokenizer and chat template"
        )
    bare_content_tokens = sum(
        len(tokenizer.encode(message.get("content", ""), add_special_tokens=False))
        for message in messages
    )
    return messages, actual, bare_content_tokens


def prepare(specification: dict[str, Any], directory: Path) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(specification["tokenizer"])
    observations = []
    failures = []
    for condition in specification["conditions"]:
        condition_directory = directory / condition["name"]
        condition_directory.mkdir(exist_ok=False)
        prepared_files = []
        condition_observations = []
        try:
            for phase, repetitions in (
                ("warmup", specification["warmup_repetitions"]),
                ("measured", specification["repetitions"]),
            ):
                destination = (
                    condition_directory / "warmup/input.jsonl"
                    if phase == "warmup"
                    else condition_directory / "input.jsonl"
                )
                rows = []
                for repetition in range(repetitions):
                    nonce = hashlib.sha256(
                        (
                            f"{specification['seed']}:{condition['name']}:{phase}:{repetition}"
                        ).encode()
                    ).hexdigest()[:32]
                    messages, actual, bare = sized_messages(
                        tokenizer,
                        condition["input_tokens"],
                        nonce,
                        specification["chat_template_kwargs"],
                    )
                    rows.append(
                        {
                            "text": "Messages supplied in extra.messages",
                            "output_length": condition["output_target"],
                            "extra": {
                                **specification["request_settings"],
                                "messages": messages,
                            },
                        }
                    )
                    condition_observations.append(
                        {
                            "condition": condition["name"],
                            "phase": phase,
                            "repetition": repetition,
                            "nonce": nonce,
                            "local_input_tokens": actual,
                            "bare_content_tokens": bare,
                            "template_and_structured_overhead_tokens": actual - bare,
                        }
                    )
                prepared_files.append((destination, rows))
        except ValueError as exc:
            failures.append({"condition": condition["name"], "diagnostic": str(exc)})
            continue
        for destination, rows in prepared_files:
            destination.parent.mkdir(exist_ok=True)
            destination.write_text("".join(json.dumps(row) + "\n" for row in rows))
        observations.extend(condition_observations)
    (directory / "preparation.json").write_text(
        json.dumps(
            {
                "tokenizer": specification["tokenizer"],
                "chat_template_kwargs": specification["chat_template_kwargs"],
                "observations": observations,
                "failures": failures,
            },
            indent=2,
        )
        + "\n"
    )


def grow(specification: dict[str, Any], directory: Path) -> None:
    """Size one growing-session turn to an exact checkpoint with accumulated history."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(specification["tokenizer"])
    history = specification["history"]
    structural = specification["structural"]
    checkpoint = specification["checkpoint"]
    tool_index = next(i for i, message in enumerate(structural) if message.get("role") == "tool")
    full_index = len(history) + tool_index
    messages: list[dict[str, Any]] = [dict(m) for m in history] + [dict(m) for m in structural]
    low, high = 0, checkpoint
    actual = -1
    found = False
    while low <= high:
        middle = (low + high) // 2
        messages[full_index]["content"] = "log" + " entry" * middle
        actual = template_count(tokenizer, messages, specification["chat_template_kwargs"])
        if actual == checkpoint:
            found = True
            break
        if actual < checkpoint:
            low = middle + 1
        else:
            high = middle - 1
    if not found:
        raise ValueError(
            f"Cannot size growing-session turn exactly to {checkpoint} tokens "
            "with the submitted tokenizer and chat template"
        )
    bare_content_tokens = sum(
        len(tokenizer.encode(message.get("content", ""), add_special_tokens=False))
        for message in messages
    )
    dataset = directory / "input.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "text": "Messages supplied in extra.messages",
                "output_length": specification["output_target"],
                "extra": {
                    **specification["request_settings"],
                    "messages": messages,
                },
            }
        )
        + "\n"
    )
    (directory / "sizing.json").write_text(
        json.dumps(
            {
                "checkpoint": checkpoint,
                "local_input_tokens": actual,
                "history_messages": len(history),
                "bare_content_tokens": bare_content_tokens,
                "template_and_structured_history_tokens": actual - bare_content_tokens,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--mode", choices=["prepare", "grow"], default="prepare")
    args = parser.parse_args()
    specification = json.loads(args.spec.read_text())
    if args.mode == "grow":
        grow(specification, args.directory)
    else:
        prepare(specification, args.directory)


if __name__ == "__main__":
    main()
