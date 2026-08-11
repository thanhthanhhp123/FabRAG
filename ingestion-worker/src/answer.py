"""Local MVP answer generation over reranked evidence with source citations."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

from .rerank import RerankedResult, retrieve_and_rerank

GENERATION_MODEL_NAME = os.environ.get("GENERATION_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

_tokenizer: Any | None = None
_model: Any | None = None

SYSTEM_PROMPT = """You answer questions about electronics-manufacturing documents.
Treat the supplied evidence as the only source of truth, even if it conflicts with
your prior knowledge. When the evidence directly states the requested value, report
that value; do not question, reinterpret, or reject it.
Use only the supplied evidence. Do not add unsupported facts.
Every factual sentence must end with one or more source IDs such as [S1].
If the evidence does not answer the question, say exactly: I don't have enough evidence.
Keep the answer concise and never invent a filename, page number, or source ID."""

EXAMPLE_USER_PROMPT = """Question: What is the operating temperature?

Evidence:

[S1] File: example.pdf; page: 2
The operating temperature is -40 C to 85 C."""
EXAMPLE_ASSISTANT_ANSWER = "The operating temperature is -40 C to 85 C [S1]."


@dataclass(frozen=True)
class GeneratedAnswer:
    question: str
    text: str
    sources: tuple[RerankedResult, ...]


def _page_label(result: RerankedResult) -> str:
    if result.page_start == result.page_end:
        return str(result.page_start)
    return f"{result.page_start}-{result.page_end}"


def build_evidence_prompt(question: str, evidence: list[RerankedResult]) -> str:
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")
    if not evidence:
        return f"Question: {question}\n\nEvidence: No relevant evidence was retrieved."

    blocks = []
    for index, result in enumerate(evidence, start=1):
        blocks.append(
            f"[S{index}] File: {result.filename}; pages: {_page_label(result)}\n{result.text}"
        )
    return f"Question: {question}\n\nEvidence:\n\n" + "\n\n".join(blocks)


def get_generator() -> tuple[Any, Any]:
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(GENERATION_MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def generate_from_evidence(question: str, evidence: list[RerankedResult]) -> GeneratedAnswer:
    """Generate a grounded answer from already selected evidence."""
    prompt = build_evidence_prompt(question, evidence)
    if not evidence:
        return GeneratedAnswer(
            question=question.strip(),
            text="I don't have enough evidence.",
            sources=(),
        )

    tokenizer, model = get_generator()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": EXAMPLE_USER_PROMPT},
        {"role": "assistant", "content": EXAMPLE_ASSISTANT_ANSWER},
        {"role": "user", "content": prompt},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors="pt")
    generated = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    input_length = inputs["input_ids"].shape[1]
    text = tokenizer.decode(generated[0][input_length:], skip_special_tokens=True).strip()
    return GeneratedAnswer(question=question.strip(), text=text, sources=tuple(evidence))


def answer_question(
    question: str,
    candidate_k: int = 10,
    top_n: int = 3,
) -> GeneratedAnswer:
    evidence = retrieve_and_rerank(question, candidate_k, top_n)
    return generate_from_evidence(question, evidence)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()

    result = answer_question(args.question, args.candidate_k, args.top_n)
    print(f"\nAnswer:\n{result.text}")
    print("\nSources:")
    for index, source in enumerate(result.sources, start=1):
        print(f"[S{index}] {source.filename}, pages {_page_label(source)}")


if __name__ == "__main__":
    main()
