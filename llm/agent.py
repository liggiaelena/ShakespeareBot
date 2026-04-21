import anthropic
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL

# ── System prompt — unchanged word for word ───────────────────────────────────
_SYSTEM = """\
You are William Shakespeare — the playwright, poet, and actor of Elizabethan England, \
speaking as though alive in the present day yet wholly true to your voice and your era.

Your persona:
- Speak in the first person as Shakespeare. Refer to your own plays, sonnets, and life \
  experiences when relevant (the Globe Theatre, your patrons, your company the Lord Chamberlain's Men, \
  your birthplace Stratford-upon-Avon, your contemporaries such as Marlowe and Jonson).
- Use early-modern English flavour naturally: occasional "thee", "thou", "dost", "hath", \
  "methinks", "prithee", "forsooth" — but never so thick that the meaning is lost.
- Draw on the reference material provided to enrich your answers with accurate historical detail.
- If asked about something wholly outside your world (modern technology, events after 1616), \
  acknowledge your ignorance with wit and humility, then offer what wisdom you can by analogy.

Rules:
- Respond in the SAME language the interlocutor used ({language}), while preserving your \
  Shakespearean voice and style as much as that language allows.
- Be concise — your answer will be read aloud as audio, so avoid bullet lists and markdown. Aim for 3-4 sentences maximum.
- Integrate context from the reference material naturally, as if recalling your own memories.
"""

# ── Tool definitions ──────────────────────────────────────────────────────────
_TOOLS = [
    {
        "name": "search_shakespeare_texts",
        "description": (
            "Search the Shakespeare knowledge base for relevant passages. "
            "Use this when the question is about Shakespeare's plays, sonnets, themes, "
            "characters, literary analysis, the Elizabethan era, theatrical history, or "
            "art history surrounding Shakespeare's world. Returns source-labelled text chunks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant Shakespeare passages.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "answer_directly",
        "description": (
            "Answer the question directly from your own knowledge as Shakespeare, "
            "without searching the knowledge base. Use this for greetings, simple "
            "conversational exchanges, biographical facts already established in the "
            "conversation history, or questions you can answer confidently from general "
            "Shakespeare knowledge."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "ask_for_clarification",
        "description": (
            "Ask the user to clarify their question. Use this when the question is "
            "too vague to search meaningfully — for example 'tell me about the play' "
            "with no play name mentioned, or 'what did he write' with no subject."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "The clarifying question to ask the user, spoken in Shakespeare's voice."
                    ),
                }
            },
            "required": ["question"],
        },
    },
]

_MAX_LOOP = 3  # cap agentic iterations to prevent runaway loops


class Agent:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def ask(
        self,
        question: str,
        search_fn,
        history: list[dict],
        language: str,
        query_type: str = "unknown",
        confidence: float = 0.0,
    ) -> str:
        """Agentic loop: Claude decides whether to search, answer directly, or clarify.

        Args:
            question:   The user's current message.
            search_fn:  Callable(query: str) -> list[str] — the retriever's search method.
            history:    Up to 6 prior messages as {"role": "user/assistant", "content": str}.
            language:   ISO 639-1 code for the reply language.

        Returns:
            Shakespeare's final text response.
        """
        # Prepend classification hint to the user turn so Claude has genre context.
        # This is NOT stored in conversation history (server.py stores plain question).
        if query_type != "unknown":
            user_content = (
                f"Query classification: {query_type} "
                f"(ML classifier confidence: {confidence:.0%})\n\n"
                f"{question}"
            )
        else:
            user_content = question

        working_messages = list(history) + [{"role": "user", "content": user_content}]

        for _ in range(_MAX_LOOP):
            response = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=320,
                system=_SYSTEM.format(language=language),
                tools=_TOOLS,
                messages=working_messages,
            )

            if response.stop_reason == "tool_use":
                tool_block = next(
                    b for b in response.content if b.type == "tool_use"
                )
                # Append assistant's tool-use turn to the working conversation
                working_messages.append(
                    {"role": "assistant", "content": response.content}
                )

                if tool_block.name == "search_shakespeare_texts":
                    chunks = search_fn(tool_block.input["query"])
                    tool_result = (
                        "\n\n".join(chunks) if chunks else "No relevant passages found."
                    )

                elif tool_block.name == "answer_directly":
                    tool_result = "Proceed to answer directly."

                else:  # ask_for_clarification
                    return tool_block.input.get(
                        "question",
                        "Prithee, couldst thou be more specific in thy question?",
                    )

                working_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_block.id,
                                "content": tool_result,
                            }
                        ],
                    }
                )

            else:  # stop_reason == "end_turn" — direct answer, no tool needed
                text_block = next(
                    (b for b in response.content if hasattr(b, "text")), None
                )
                if text_block:
                    return text_block.text
                break

        return "I am at a loss for words, good friend. Prithee, ask again."
