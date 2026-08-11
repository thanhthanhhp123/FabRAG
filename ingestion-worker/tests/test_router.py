import pytest

from src.router import Route, RouteDecision, parse_route_decision, route_question


def test_parse_single_hop_decision():
    decision = parse_route_decision('{"route":"single_hop","subqueries":[]}', "voltage?")

    assert decision == RouteDecision(Route.SINGLE_HOP)


def test_parse_multi_hop_decision():
    decision = parse_route_decision(
        '{"route":"multi_hop","subqueries":["LM317 output current","LM358 current"]}',
        "compare",
    )

    assert decision == RouteDecision(Route.MULTI_HOP, ("LM317 output current", "LM358 current"))


def test_parse_fenced_multi_hop_objects_ignores_model_answers():
    decision = parse_route_decision(
        '```json\n{"route":"multi_hop","subqueries":['
        '{"query":"LM317 voltage","answer":"invented"},'
        '{"query":"DRV8833 voltage"}]}\n```',
        "compare",
    )

    assert decision == RouteDecision(Route.MULTI_HOP, ("LM317 voltage", "DRV8833 voltage"))


@pytest.mark.parametrize(
    "output",
    [
        "not json",
        "[]",
        '{"route":"unknown","subqueries":[]}',
        '{"route":"multi_hop","subqueries":["only one"]}',
        '{"route":"multi_hop","subqueries":"wrong type"}',
    ],
)
def test_invalid_decision_falls_back_to_original_question(output):
    assert parse_route_decision(output, "  original question  ") == RouteDecision(
        Route.SINGLE_HOP, ("original question",)
    )


def test_route_question_rejects_blank_input():
    with pytest.raises(ValueError, match="question"):
        route_question(" ", lambda _: "unused")


def test_route_question_uses_injected_generator():
    assert route_question(
        "How do transistors work?",
        lambda question: '{"route":"general_knowledge","subqueries":[]}',
    ) == RouteDecision(Route.GENERAL_KNOWLEDGE)


def test_route_question_rejects_obvious_out_of_domain_before_model():
    assert route_question(
        "Who won the football World Cup?",
        lambda _: pytest.fail("model should not run for an obvious rejection"),
    ) == RouteDecision(Route.REJECT)


def test_out_of_domain_word_does_not_override_electronics_context():
    assert route_question(
        "How can I show a football score on a microcontroller display?",
        lambda _: '{"route":"single_hop","subqueries":[]}',
    ) == RouteDecision(Route.SINGLE_HOP)
