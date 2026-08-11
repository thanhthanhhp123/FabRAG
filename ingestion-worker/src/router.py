"""LLM-backed query routing with strict validation and a safe retrieval fallback."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum


class Route(StrEnum):
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    GENERAL_KNOWLEDGE = "general_knowledge"
    REJECT = "reject"


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    subqueries: tuple[str, ...] = ()


ROUTER_SYSTEM_PROMPT = """Route an incoming question for an electronics-document assistant.
Return only one JSON object with keys route and subqueries.
route must be one of: single_hop, multi_hop, general_knowledge, reject.
- single_hop: one document lookup can answer it.
- multi_hop: comparison or synthesis needs two or more separate lookups.
- general_knowledge: in-domain electronics knowledge that does not require a datasheet.
- reject: unrelated to electronics, manufacturing, components, or engineering.
For multi_hop, provide 2 to 4 standalone retrieval subqueries. Otherwise use [].
Do not answer the question and do not add markdown."""

ROUTER_EXAMPLES = (
    (
        "Compare the supply voltage ranges of the L293 and DRV8833.",
        (
            '{"route":"multi_hop","subqueries":["L293 supply voltage range",'
            '"DRV8833 supply voltage range"]}'
        ),
    ),
    (
        "Who won the football World Cup?",
        '{"route":"reject","subqueries":[]}',
    ),
)

DOMAIN_TERMS = frozenset(
    {
        "adc",
        "amplifier",
        "circuit",
        "component",
        "current",
        "datasheet",
        "diode",
        "electronics",
        "gpio",
        "manufacturing",
        "microcontroller",
        "op-amp",
        "pcb",
        "register",
        "regulator",
        "resistor",
        "sensor",
        "supply",
        "transistor",
        "voltage",
    }
)
OUT_OF_DOMAIN_TERMS = frozenset(
    {
        "basketball",
        "celebrity",
        "cinema",
        "football",
        "movie",
        "recipe",
        "soccer",
        "sports",
        "world cup",
    }
)


def _obviously_out_of_domain(question: str) -> bool:
    normalized = question.casefold()
    has_domain_term = any(term in normalized for term in DOMAIN_TERMS)
    has_out_of_domain_term = any(term in normalized for term in OUT_OF_DOMAIN_TERMS)
    return has_out_of_domain_term and not has_domain_term


def parse_route_decision(raw_output: str, question: str) -> RouteDecision:
    """Validate router JSON; malformed or unsafe output falls back to retrieval."""
    fallback = RouteDecision(Route.SINGLE_HOP, (question.strip(),))
    cleaned = raw_output.strip()
    object_start = cleaned.find("{")
    object_end = cleaned.rfind("}")
    if object_start >= 0 and object_end > object_start:
        cleaned = cleaned[object_start : object_end + 1]
    try:
        payload = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return fallback
    if not isinstance(payload, dict):
        return fallback
    try:
        route = Route(payload.get("route"))
    except (ValueError, TypeError):
        return fallback

    raw_subqueries = payload.get("subqueries", [])
    if not isinstance(raw_subqueries, list):
        return fallback
    normalized_subqueries: list[str] = []
    for item in raw_subqueries:
        if isinstance(item, dict):
            item = item.get("query")
        if isinstance(item, str) and item.strip():
            normalized_subqueries.append(item.strip()[:500])
    subqueries = tuple(normalized_subqueries)
    if route is Route.MULTI_HOP:
        if not 2 <= len(subqueries) <= 4:
            return fallback
        return RouteDecision(route, subqueries)
    return RouteDecision(route)


def _generate_router_output(question: str) -> str:
    from .answer import get_generator

    tokenizer, model = get_generator()
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
    ]
    for example_question, example_answer in ROUTER_EXAMPLES:
        messages.extend(
            [
                {"role": "user", "content": example_question},
                {"role": "assistant", "content": example_answer},
            ]
        )
    messages.append({"role": "user", "content": question})
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors="pt")
    if hasattr(inputs, "to") and hasattr(model, "device"):
        inputs = inputs.to(model.device)
    generated = model.generate(
        **inputs,
        max_new_tokens=160,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    input_length = inputs["input_ids"].shape[1]
    return tokenizer.decode(generated[0][input_length:], skip_special_tokens=True).strip()


def route_question(
    question: str,
    generate_fn: Callable[[str], str] = _generate_router_output,
) -> RouteDecision:
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")
    if _obviously_out_of_domain(question):
        return RouteDecision(Route.REJECT)
    return parse_route_decision(generate_fn(question), question)
