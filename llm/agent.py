import anthropic
from anthropic.types import TextBlock
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL

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
- Be concise — your answer will be read aloud as audio, so avoid bullet lists and markdown.
- Integrate context from the reference material naturally, as if recalling your own memories.
"""


class Agent:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def ask(self, question: str, context: list[str], language: str) -> str:
        """Gera uma resposta na voz de Shakespeare, contextualizada em `language`."""
        context_block = "\n\n".join(context) if context else "No reference material available."

        user_message = (
            f"Interlocutor's question: {question}\n\n"
            f"Reference material from the Shakespeare knowledge base:\n{context_block}\n\n"
            "Please answer in character as William Shakespeare, drawing on the reference material."
        )

        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=512,
            system=_SYSTEM.format(language=language),
            messages=[{"role": "user", "content": user_message}],
        )

        text_block = next(b for b in response.content if isinstance(b, TextBlock))
        return text_block.text
