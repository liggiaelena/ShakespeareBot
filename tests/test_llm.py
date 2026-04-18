"""Tests the Agent with a mocked Anthropic API to avoid consuming credits."""
from unittest.mock import MagicMock, patch
from llm.agent import Agent


def _make_mock_response(text: str):
    """Simulates a direct end_turn response (no tool use)."""
    content_block = MagicMock()
    content_block.text = text
    response = MagicMock()
    response.content = [content_block]
    response.stop_reason = "end_turn"
    return response


def test_agent_returns_string():
    agent = Agent()
    mock_response = _make_mock_response(
        "Methinks thou dost ask of Hamlet — 'tis a tale of a Danish prince consumed by doubt."
    )

    def mock_search(query: str) -> list:
        return ["Hamlet is a tragedy written by Shakespeare around 1600."]

    with patch.object(agent.client.messages, "create", return_value=mock_response):
        result = agent.ask(
            question="Tell me about Hamlet.",
            search_fn=mock_search,
            history=[],
            language="en",
        )

    assert isinstance(result, str)
    assert len(result) > 0


def test_agent_passes_language_in_system_prompt():
    agent = Agent()
    mock_response = _make_mock_response("Hamlet es un príncipe danés atormentado por la duda.")
    captured = {}

    def capture_call(**kwargs):
        captured.update(kwargs)
        return mock_response

    with patch.object(agent.client.messages, "create", side_effect=capture_call):
        agent.ask(
            question="¿De qué trata Hamlet?",
            search_fn=lambda q: [],
            history=[],
            language="es",
        )

    assert "es" in captured["system"]


def test_agent_includes_history_in_messages():
    """History should appear before the current user message."""
    agent = Agent()
    mock_response = _make_mock_response("His father was King Hamlet, murdered by Claudius.")
    captured = {}

    def capture_call(**kwargs):
        captured.update(kwargs)
        return mock_response

    history = [
        {"role": "user", "content": "Tell me about Hamlet."},
        {"role": "assistant", "content": "Hamlet is a Prince of Denmark consumed by doubt."},
    ]

    with patch.object(agent.client.messages, "create", side_effect=capture_call):
        agent.ask(
            question="And his father?",
            search_fn=lambda q: [],
            history=history,
            language="en",
        )

    messages = captured["messages"]
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Tell me about Hamlet."
    assert messages[-1]["content"] == "And his father?"
