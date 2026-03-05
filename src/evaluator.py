from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .answerer import AnswerGenerator
from .embeddings import EmbeddingEncoder
from .guardrails import validate
from .retriever import Retriever
from .utils import load_json
from .vector_store import VectorStore


@dataclass
class EvalRow:
    question: str
    expected_keywords: list[str]


def run_eval(questions_path: Path, top_k: int = 6) -> None:
    payload = load_json(questions_path)
    rows = [EvalRow(**row) for row in payload["questions"]]

    encoder = EmbeddingEncoder()
    store = VectorStore()
    retriever = Retriever(store=store, encoder=encoder)
    answerer = AnswerGenerator()

    correct = 0
    for i, row in enumerate(rows, start=1):
        chunks, scores = retriever.search(row.question, top_k=top_k)
        raw = answerer.generate_answer(row.question, chunks, chat_history=[])
        final = validate(raw.answer, chunks, scores)
        answer = final.answer.lower()
        matched = all(keyword.lower() in answer for keyword in row.expected_keywords)
        correct += int(matched)
        print(f"[{i}] Q: {row.question}")
        print(f"    A: {final.answer}")
        print(f"    Confidence: {final.confidence} | Match: {matched}")

    total = len(rows)
    print(f"\nAccuracy proxy: {correct}/{total} ({(100.0 * correct / max(1, total)):.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple local QA evaluation.")
    parser.add_argument(
        "--questions",
        required=True,
        type=Path,
        help="Path to JSON with format: {\"questions\": [{\"question\":..., \"expected_keywords\": [...]}]}",
    )
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()
    run_eval(args.questions, top_k=args.top_k)


if __name__ == "__main__":
    main()

