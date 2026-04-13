"""Tests the Agent with a mocked Anthropic API to avoid consuming credits."""
from unittest.mock import MagicMock, patch
from llm.agent import Agent


def _make_mock_response(text: str):
    content_block = MagicMock()
    content_block.text = text
    response = MagicMock()
    response.content = [content_block]
    return response


def test_agent_returns_string():
    agent = Agent()
    mock_response = _make_mock_response(
        "Methinks thou dost ask of Hamlet — 'tis a tale of a Danish prince consumed by doubt."
    )

    with patch.object(agent.client.messages, "create", return_value=mock_response):
        result = agent.ask(
            question="Tell me about Hamlet.",
            context=["Hamlet is a tragedy written by Shakespeare around 1600."],
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
            context=[],
            language="es",
        )

    assert "es" in captured["system"]
