from __future__ import annotations

from ollama import Client
from ollama import ResponseError

from . import config
from .types import AnswerResult, Chunk


SYSTEM_PROMPT = """You are a study assistant.
Rules:
1) Answer only using the provided context.
2) If context is insufficient, say exactly: "I don't have enough information in the provided notes."
3) Keep answers concise and factual.
4) Do not use outside knowledge.
5) If user asks for a neat diagram/flowchart, return Mermaid syntax in a fenced ```mermaid block.
6) For diagrams, use only entities/relationships present in the provided context.
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
        user_prompt = (
            f"{history_block}\n"
            f"Question: {query}\n\n"
            f"Context:\n{context_block}\n\n"
            f"{_diagram_instruction(query)}\n"
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
            "1) one short explanation line, then\n"
            "2) a fenced ```mermaid code block."
        )
    return ""


class LLMRuntimeError(RuntimeError):
    pass


def _is_oom_error(exc: ResponseError) -> bool:
    text = str(exc).lower()
    return ("out of memory" in text) or ("cudamalloc failed" in text) or ("runner process has terminated" in text)
