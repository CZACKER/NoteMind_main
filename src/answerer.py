from __future__ import annotations
import re

from ollama import Client
from ollama import ResponseError

from . import config
from .types import AnswerResult, Chunk


SYSTEM_PROMPT = """You are a study assistant.
Rules:
1) Answer only using the provided context.
2) If context is insufficient, say exactly: "I don't have enough information in the provided notes."
3) Keep answers concise, factual, and exam-ready.
4) Do not use outside knowledge.
5) Default answer format must be systematic and exam-ready.
6) If user asks for a diagram/flowchart/neat diagram:
   - Prefer a clear textual explanation of the diagram structure from notes.
   - Do NOT output Mermaid.
   - Only draw a small text diagram if absolutely needed.
7) For diagrams, use only entities/relationships present in the provided context.
"""


class AnswerGenerator:
    def __init__(
        self,
        model_name: str = config.OLLAMA_MODEL_NAME,
        host: str = config.OLLAMA_HOST,
        timeout_s: int = config.OLLAMA_TIMEOUT_S,
    ) -> None:
        self.model_name = model_name
        self.client = Client(host=host, timeout=timeout_s)

    def generate_answer(
        self,
        query: str,
        contexts: list[Chunk],
        chat_history: list[dict[str, str]],
        num_predict: int = 350,
    ) -> AnswerResult:
        context_block = _build_context_block(contexts)
        history_block = _build_history_block(chat_history)
        diagram_hint = _diagram_grounding_hint(query, contexts)
        user_prompt = (
            f"{history_block}\n"
            f"Question: {query}\n\n"
            f"Context:\n{context_block}\n\n"
            f"{_answer_structure_instruction(query)}\n"
            f"{_diagram_instruction(query)}\n"
            f"{diagram_hint}\n"
            "Return only the final answer text."
        )

        base_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        base_opts = {
            "temperature": 0.0,
            "top_p": 0.9,
            "num_predict": num_predict,
            "num_ctx": config.OLLAMA_NUM_CTX,
            "seed": 7,
        }

        try:
            response = self.client.chat(model=self.model_name, options=base_opts, messages=base_messages)
            text = response["message"]["content"].strip()
            if _needs_diagram_retry(query, text):
                text = self._retry_strict_diagram_answer(
                    user_prompt=user_prompt,
                    diagram_hint=diagram_hint,
                    num_predict=num_predict,
                )
            return AnswerResult(answer=text, model=self.model_name, raw_response=text)
        except ResponseError as exc:
            if not _is_oom_error(exc):
                raise

        # OOM fallback attempt 1: reduce tokens and context + fewer chunks.
        reduced_context = contexts[: max(2, min(4, len(contexts)))]
        reduced_prompt = (
            f"{history_block}\n"
            f"Question: {query}\n\n"
            f"Context:\n{_build_context_block(reduced_context)}\n\n"
            f"{_answer_structure_instruction(query)}\n"
            f"{_diagram_instruction(query)}\n"
            "Return only the final answer text."
        )
        reduced_opts = {
            "temperature": 0.0,
            "top_p": 0.9,
            "num_predict": min(160, num_predict),
            "num_ctx": min(1024, config.OLLAMA_NUM_CTX),
            "seed": 7,
        }
        try:
            response = self.client.chat(
                model=self.model_name,
                options=reduced_opts,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": reduced_prompt},
                ],
            )
            text = response["message"]["content"].strip()
            return AnswerResult(answer=text, model=self.model_name, raw_response=text)
        except ResponseError as exc2:
            if not _is_oom_error(exc2):
                raise

        # OOM fallback attempt 2: smaller model (if pulled locally).
        if config.OLLAMA_FALLBACK_MODEL:
            try:
                response = self.client.chat(
                    model=config.OLLAMA_FALLBACK_MODEL,
                    options=reduced_opts,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": reduced_prompt},
                    ],
                )
                text = response["message"]["content"].strip()
                return AnswerResult(
                    answer=text,
                    model=config.OLLAMA_FALLBACK_MODEL,
                    raw_response=text,
                )
            except ResponseError as exc3:
                if _is_oom_error(exc3):
                    raise LLMRuntimeError(
                        "GPU out of memory in Ollama. Try a smaller model (for example "
                        "'ollama pull qwen2.5:3b-instruct'), close other GPU apps, or use CPU mode."
                    ) from exc3
                raise

        raise LLMRuntimeError(
            "GPU out of memory in Ollama. Reduce model size or available context."
        )


def _build_context_block(contexts: list[Chunk]) -> str:
    if not contexts:
        return "[NO CONTEXT]"
    lines: list[str] = []
    for i, chunk in enumerate(contexts, start=1):
        lines.append(
            f"[{i}] file={chunk.filename} page={chunk.page_num} chunk_id={chunk.chunk_id}\n{chunk.text}"
        )
    return "\n\n".join(lines)


def _build_history_block(chat_history: list[dict[str, str]]) -> str:
    if not chat_history:
        return ""
    trimmed = chat_history[-4:]
    lines = ["Conversation history:"]
    for message in trimmed:
        role = message["role"].capitalize()
        lines.append(f"{role}: {message['content']}")
    return "\n".join(lines)


def _diagram_instruction(query: str) -> str:
    q = query.lower()
    if any(word in q for word in ["diagram", "flowchart", "draw", "neat diagram", "chart"]):
        return (
            "User requests diagram output. If context is sufficient, respond with:\n"
            "1) one short explanation line,\n"
            "2) then exact diagram flow described using original node names from notes,\n"
            "3) then 2-4 interpretation points.\n"
            "IMPORTANT: Do not invent placeholders like Activity 1/2/3 unless those exact words are in context."
        )
    return ""


def _answer_structure_instruction(query: str) -> str:
    if _is_diagram_query(query):
        return (
            "Use EXACT sections in this order with markdown headings:\n"
            "### Definition\n"
            "### Detailed Explanation\n"
            "### Diagram Explanation\n"
            "### Example\n"
            "### Exam Answer\n"
            "Rules:\n"
            "- In Diagram Explanation, explain the flow node-by-node using exact names from notes.\n"
            "- Example must be from notes/context only.\n"
            "- Exam Answer should be 8-12 lines and ready to write in exam."
        )
    return (
        "Use EXACT sections in this order with markdown headings:\n"
        "### Definition\n"
        "### Detailed Explanation\n"
        "### Example\n"
        "### Exam Answer\n"
        "Rules:\n"
        "- Detailed Explanation should use numbered points.\n"
        "- Exam Answer should be 8-12 lines and ready to write in exam."
    )


class LLMRuntimeError(RuntimeError):
    pass


def _is_oom_error(exc: ResponseError) -> bool:
    text = str(exc).lower()
    return ("out of memory" in text) or ("cudamalloc failed" in text) or ("runner process has terminated" in text)


def _needs_diagram_retry(query: str, answer_text: str) -> bool:
    if not _is_diagram_query(query):
        return False
    text = answer_text.lower()
    generic_tokens = ["activity 1", "activity 2", "activity 3", "project start", "project end"]
    return sum(1 for t in generic_tokens if t in text) >= 2


def _is_diagram_query(query: str) -> bool:
    q = query.lower()
    return any(word in q for word in ["diagram", "flowchart", "draw", "neat diagram", "chart"])


def _diagram_grounding_hint(query: str, contexts: list[Chunk]) -> str:
    if not _is_diagram_query(query):
        return ""
    joined = "\n".join(chunk.text for chunk in contexts[:6])
    candidates: list[str] = []

    patterns = [
        r"\bdefine\s+the\s+overall\s+system\b",
        r"\bselect\s+module\s+[a-z]\b",
        r"\bdesign\s+module\s+[a-z]\b",
        r"\bmodule\s+[a-z]\b",
        r"\bcode\s*/\s*test\b",
        r"\bintegrate\b",
        r"\btest\s+system\b",
        r"\bcheck\s+specifications\b",
    ]
    lower = joined.lower()
    for pat in patterns:
        for m in re.finditer(pat, lower):
            phrase = lower[m.start() : m.end()].strip()
            if phrase and phrase not in candidates:
                candidates.append(phrase)

    if not candidates:
        return (
            "If the context does not provide clear diagram node names, do not invent diagram nodes. "
            'Instead say exactly: "I don\'t have enough information in the provided notes."'
        )
    terms = ", ".join(candidates[:20])
    return (
        "Diagram grounding terms found in notes: "
        f"{terms}. Use these exact terms while describing the diagram."
    )


def _retry_prompt(user_prompt: str, diagram_hint: str) -> str:
    return (
        f"{user_prompt}\n\n"
        "Retry with strict grounding:\n"
        "- Use exact node labels from context/grounding terms.\n"
        "- Do not use generic placeholders.\n"
        f"- Grounding terms reminder: {diagram_hint}\n"
    )


def _safe_num_predict(n: int) -> int:
    return max(96, min(220, n))


def _retry_options(num_predict: int) -> dict:
    return {
        "temperature": 0.0,
        "top_p": 0.9,
        "num_predict": _safe_num_predict(num_predict),
        "num_ctx": min(1200, config.OLLAMA_NUM_CTX),
        "seed": 7,
    }


def _strict_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _retry_strict_diagram_answer_impl(
    client: Client,
    model: str,
    user_prompt: str,
    diagram_hint: str,
    num_predict: int,
) -> str:
    response = client.chat(
        model=model,
        options=_retry_options(num_predict),
        messages=_strict_messages(_retry_prompt(user_prompt, diagram_hint)),
    )
    return response["message"]["content"].strip()


def _retry_strict_diagram_answer_fallback(
    client: Client,
    user_prompt: str,
    diagram_hint: str,
    num_predict: int,
) -> str:
    response = client.chat(
        model=config.OLLAMA_FALLBACK_MODEL or config.OLLAMA_MODEL_NAME,
        options=_retry_options(num_predict),
        messages=_strict_messages(_retry_prompt(user_prompt, diagram_hint)),
    )
    return response["message"]["content"].strip()


def _retry_wrapper(
    client: Client,
    model: str,
    user_prompt: str,
    diagram_hint: str,
    num_predict: int,
) -> str:
    try:
        return _retry_strict_diagram_answer_impl(client, model, user_prompt, diagram_hint, num_predict)
    except ResponseError:
        return _retry_strict_diagram_answer_fallback(client, user_prompt, diagram_hint, num_predict)


def _bind_retry_method() -> None:
    def _retry_strict_diagram_answer(self, user_prompt: str, diagram_hint: str, num_predict: int) -> str:
        return _retry_wrapper(self.client, self.model_name, user_prompt, diagram_hint, num_predict)

    setattr(AnswerGenerator, "_retry_strict_diagram_answer", _retry_strict_diagram_answer)


_bind_retry_method()
