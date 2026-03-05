from __future__ import annotations


class ConversationMemory:
    def __init__(self) -> None:
        self._messages: list[dict[str, str]] = []

    def add_user(self, message: str) -> None:
        self._messages.append({"role": "user", "content": message})

    def add_assistant(self, message: str) -> None:
        self._messages.append({"role": "assistant", "content": message})

    def get(self) -> list[dict[str, str]]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

