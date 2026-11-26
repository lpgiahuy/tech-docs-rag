"""Evaluate retrieval (Recall@k, MRR, latency) and, optionally, the generated answers.

Usage:
    python eval/run_eval.py                      # retrieval only, no LLM calls
    python eval/run_eval.py --mode hybrid --rerank
    python eval/run_eval.py --generate --resume  # also answer the questions with the LLM
    python eval/run_eval.py --grades             # summarize the manual grades
"""

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings
from app.generator import REFUSAL, LLMError, filter_relevant, generate
from app.ingest import embed
from app.retriever import search

EVAL_SET = ROOT / "eval" / "eval_set.json"
ANSWERS = ROOT / "eval" / "answers.json"
KS = tuple(sorted({1, 3, 5, 10, settings.top_k}))
DEPTH = max(KS)
CATEGORIES = ("exact", "paraphrase", "multi", "unanswerable")
GRADES = ("correct", "partial", "wrong", "correct_refusal")


def first_relevant_rank(chunks: list[dict], item: dict) -> int | None:
    """1-based rank of the first chunk from a relevant page, or None."""
    pages = {item["relevant_page"], *item.get("also_relevant_pages", [])}
    for rank, chunk in enumerate(chunks, start=1):
        if chunk["filename"] == item["relevant_source"] and chunk["page"] in pages:
            return rank
    return None


def recall_at(ranks: list[int | None], k: int) -> float:
    return sum(r is not None and r <= k for r in ranks) / len(ranks)


def mrr(ranks: list[int | None]) -> float:
    return sum(1 / r for r in ranks if r is not None) / len(ranks)


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile: the smallest value with at least q% of values at or below it."""
    ordered = sorted(values)
    return ordered[max(math.ceil(q / 100 * len(ordered)) - 1, 0)]


def retrieve(items: list[dict]) -> list[dict]:
    """Search every question once. Each result holds the chunks, rank and latency."""
    embed(["warm up"])
    search("warm up", k=DEPTH)
    results = []
    for item in items:
        start = time.perf_counter()
        chunks = search(item["question"], k=DEPTH)
        elapsed_ms = (time.perf_counter() - start) * 1000
        rank = first_relevant_rank(chunks, item) if item["relevant_source"] else None
        results.append({"item": item, "chunks": chunks, "rank": rank, "ms": elapsed_ms})
    return results


def print_retrieval(results: list[dict]) -> None:
    print("\nPer question (rank of first relevant chunk, '-' = not in top %d)" % DEPTH)
    for r in results:
        item = r["item"]
        rank = "n/a" if not item["relevant_source"] else (r["rank"] or "-")
        print(
            f"  {item['id']}  {item['category']:<12} rank={rank!s:<4} "
            f"top score={r['chunks'][0]['score']:.3f}  {r['ms']:6.0f} ms"
        )

    header = "".join(f"{f'R@{k}':>8}" for k in KS)
    print(f"\n{'category':<12}{'n':>4}{header}{'MRR':>8}{'avg ms':>9}{'p95 ms':>9}")
    answerable = [r for r in results if r["item"]["relevant_source"]]
    groups = [(c, [r for r in answerable if r["item"]["category"] == c]) for c in CATEGORIES]
    for name, group in [*groups, ("ALL", answerable)]:
        if not group:
            continue
        ranks = [r["rank"] for r in group]
        times = [r["ms"] for r in group]
        recalls = "".join(f"{recall_at(ranks, k):8.2f}" for k in KS)
        print(
            f"{name:<12}{len(group):>4}{recalls}{mrr(ranks):8.2f}"
            f"{sum(times) / len(times):9.0f}{percentile(times, 95):9.0f}"
        )

    times = [r["ms"] for r in results]
    print(
        f"\nRetrieval latency over all {len(times)} questions: "
        f"avg {sum(times) / len(times):.0f} ms, p95 {percentile(times, 95):.0f} ms"
    )
    unanswerable = [r for r in results if not r["item"]["relevant_source"]]
    blocked = sum(not filter_relevant(r["chunks"][: settings.top_k]) for r in unanswerable)
    print(
        f"Unanswerable questions stopped by min_score={settings.min_score} alone: "
        f"{blocked}/{len(unanswerable)}"
    )


def load_answers() -> dict[str, dict]:
    if not ANSWERS.exists():
        return {}
    return {a["id"]: a for a in json.loads(ANSWERS.read_text(encoding="utf-8"))}


def save_answers(answers: dict[str, dict]) -> None:
    text = json.dumps([answers[i] for i in sorted(answers)], indent=2, ensure_ascii=False)
    ANSWERS.write_text(text + "\n", encoding="utf-8", newline="\n")


def generate_with_retry(question: str, chunks: list[dict], attempts: int = 4) -> tuple[str, float]:
    """Return (answer, latency_ms of the successful call). Waits out quota/overload errors."""
    for attempt in range(1, attempts + 1):
        start = time.perf_counter()
        try:
            return generate(question, chunks), (time.perf_counter() - start) * 1000
        except LLMError as exc:
            if attempt == attempts or "PerDay" in str(exc):  # a daily quota will not clear
                raise
            print(f"LLM error, retrying in {20 * attempt}s: {str(exc)[:120]}", flush=True)
            time.sleep(20 * attempt)


def run_generation(results: list[dict], pause: float, resume: bool) -> None:
    """Answer every question as /ask does and save the answers. A grade survives a rerun
    only if the answer text is unchanged."""
    old = load_answers()
    answers = dict(old)
    print("\nGenerating answers...", flush=True)
    for r in results:
        item = r["item"]
        if resume and old.get(item["id"], {}).get("question") == item["question"]:
            continue
        try:
            answer, llm_ms = generate_with_retry(item["question"], r["chunks"][: settings.top_k])
        except LLMError as exc:
            sys.exit(f"Stopped at {item['id']}: {str(exc)[:200]}\nRerun later with --resume.")
        total_ms = r["ms"] + llm_ms
        previous = old.get(item["id"], {})
        same = previous.get("answer") == answer and previous.get("question") == item["question"]
        answers[item["id"]] = {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "answer": answer,
            "refused": answer == REFUSAL,
            "retrieval_rank": r["rank"],
            "total_ms": round(total_ms),
            "model": settings.llm_model,
            "grade": previous.get("grade") if same else None,
            "note": previous.get("note", "") if same else "",
        }
        save_answers(answers)
        print(f"  {item['id']} {round(total_ms):6d} ms  {answer[:90]!r}", flush=True)
        time.sleep(pause)
    times = [answers[r["item"]["id"]]["total_ms"] for r in results]
    print(
        f"\nEnd-to-end latency (retrieval + LLM): avg {sum(times) / len(times):.0f} ms, "
        f"p95 {percentile(times, 95):.0f} ms"
    )
    print(f"Answers saved to {ANSWERS.relative_to(ROOT)}. Fill in 'grade' with one of {GRADES}.")


def print_grades(items: list[dict]) -> None:
    answers = load_answers()
    if not answers:
        sys.exit("no answers yet: run with --generate first")
    print(f"\n{'category':<14}{'n':>4}" + "".join(f"{g:>17}" for g in (*GRADES, "ungraded")))
    for name in (*CATEGORIES, "ALL"):
        group = [
            answers.get(i["id"], {}) for i in items if name == "ALL" or i["category"] == name
        ]
        counts = Counter(a.get("grade") or "ungraded" for a in group)
        unknown = set(counts) - {*GRADES, "ungraded"}
        if unknown:
            sys.exit(f"unknown grade(s) in {ANSWERS.name}: {unknown}")
        print(
            f"{name:<14}{len(group):>4}"
            + "".join(f"{counts[g]:>17}" for g in (*GRADES, "ungraded"))
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--generate", action="store_true", help="also call the LLM")
    parser.add_argument("--resume", action="store_true", help="skip already answered questions")
    parser.add_argument("--grades", action="store_true", help="summarize manual grades")
    parser.add_argument("--model", help="evaluate this model instead of LLM_MODEL")
    parser.add_argument("--pause", type=float, default=5.0, help="seconds between LLM calls")
    args = parser.parse_args()

    settings.llm_fallback_models = ""
    if args.model:
        global ANSWERS
        settings.llm_model = args.model
        ANSWERS = ROOT / "eval" / f"answers_{args.model}.json"

    items = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    if args.grades:
        print_grades(items)
        return
    results = retrieve(items)
    print_retrieval(results)
    if args.generate:
        run_generation(results, args.pause, args.resume)


if __name__ == "__main__":
    main()
