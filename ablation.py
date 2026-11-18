"""Task 5: the ablation study.

Runs the whole pipeline once per configuration on the public test set, changing one component at a
time, and writes ablation_results.csv. The default configuration in hw4.CONFIG is the final system;
every row below differs from the row above it by exactly one component.

    python3 ablation.py --data-folder public_test
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import hw4

FINAL = dict(hw4.CONFIG)

# Each row: label, the settings that differ from the row above, and why the change was tried.
OFF = dict(table_to_text=False, structured_chunking=False, metadata_filter=False, hybrid_scoring=False,
           context_header=False, better_tokenizer=False, per_company_retrieval=False, abstention=False,
           compact_context=False)

STEPS = [
    ("B0 naive RAG", OFF,
     "get_text() + RecursiveCharacterTextSplitter(1000, 150), required metadata, rank_bm25 top-5 over "
     "the whole corpus with lower-cased whitespace tokens, no query parsing, never []."),
    ("V1 table_to_text", dict(table_to_text=True),
     "Each table becomes one line per row with its label and column headers; still flat-split."),
    ("V2 structured chunking", dict(structured_chunking=True),
     "One chunk per table (split on row boundaries), narrative grouped by section instead of a "
     "fixed-size character splitter."),
    ("V3 metadata filtering", dict(metadata_filter=True),
     "parse_query(); the BM25 pool is restricted to the question's company and document type."),
    ("V4 hybrid scoring", dict(hybrid_scoring=True),
     "BM25 + a year bonus + IDF-weighted share of question terms in the chunk + a table-row bonus."),
    ("V5 context header", dict(context_header=True),
     "Every chunk starts with company | FY | doc type | caption | section."),
    ("V6 better tokenizer", dict(better_tokenizer=True),
     "Stop words, possessives, numbers without commas, plural folding, synonym expansion."),
    ("V7 per-company retrieval", dict(per_company_retrieval=True),
     "A comparison runs one ranking per company and takes the top chunks of each."),
    ("V8 abstention", dict(abstention=True),
     "[] for a company not in the corpus, a year its documents do not state, an undisclosed-fact "
     "phrase, or a failed model YES/NO check when lexical coverage is low."),
    ("V9 compact context", dict(compact_context=True),
     "Dedupe by table/section, 3 chunks (2 per company for a comparison), drop the tail below 35% "
     "of the best score."),
]


def run_config(corpus: dict[str, str], questions: dict[str, str], ground_truth: dict | None) -> dict:
    """Score one configuration: the same steps the runner takes, without the provided main()."""
    hw4._ENTITIES["companies"].clear()
    hw4._ENTITIES["brokers"].clear()
    hw4._ENTITIES["auditors"].clear()
    hw4._ENTITIES["docs"].clear()

    chunks_by_file = {name: hw4.chunk_document(name, html) for name, html in corpus.items()}
    all_chunks = [c for name in sorted(chunks_by_file) for c in chunks_by_file[name]
                  if len(c.page_content) <= hw4.MAX_CHUNK_CHARS]
    index = hw4.build_index(all_chunks)

    retrieved: dict[str, list] = {}
    parsed: dict[str, dict] = {}
    for qid, question in questions.items():
        parsed[qid] = hw4.parse_query(question)
        try:
            result = hw4.retrieve(index, question)
        except Exception:
            result = []
        retrieved[qid] = list(result or [])[:hw4.MAX_RETRIEVED]

    answers = {qid: hw4.generate(questions[qid], retrieved[qid]) for qid in questions}

    if not ground_truth:
        return {}
    answerable = [q for q in questions if ground_truth.get(q, {}).get("evidence")]
    unanswerable = [q for q in questions if ground_truth.get(q) and not ground_truth[q]["evidence"]]
    n_ans = max(len(answerable), 1)
    n_un = max(len(unanswerable), 1)
    hits = sum(hw4.retrieval_hit(retrieved[q], ground_truth[q]) for q in answerable) / n_ans
    kept = sum(bool(retrieved[q]) for q in answerable) / n_ans
    rejected = sum(not retrieved[q] for q in unanswerable) / n_un
    parse = sum(hw4.parse_score(parsed[q], ground_truth[q]) for q in questions) / max(len(questions), 1)
    accuracy = sum(hw4.answer_correct(answers[q], ground_truth[q]) for q in questions) / max(len(questions), 1)

    table_note = ""
    if not hw4.CONFIG["table_to_text"]:
        table_note = "n/a (naive get_text)"

    avg_chunks = sum(len(retrieved[q]) for q in questions) / max(len(questions), 1)

    return {
        "parse": parse, "recall": hits, "kept": kept, "rejected": rejected, "accuracy": accuracy,
        "avg_chunks": avg_chunks,
        "points_2_2": hw4.POINTS["2.2"] * parse,
        "points_2_3": hw4.POINTS["2.3"] * hw4.clip((hits - hw4.RECALL_ZERO) / (hw4.RECALL_FULL - hw4.RECALL_ZERO)),
        "points_3": hw4.POINTS["3"] * hw4.clip(kept + rejected - 1),
        "points_4": hw4.POINTS["4"] * accuracy,
        "table_note": table_note,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FTEC5660 HW4 ablation study")
    parser.add_argument("--data-folder", default="public_test")
    parser.add_argument("--output", default="ablation_results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = hw4.load_corpus(folder / "corpus")
    questions = hw4.load_json(folder / "questions.json")
    ground_truth = hw4.load_json(folder / "ground_truth.json")

    rows = []
    settings: dict = {}
    for label, changes, reason in STEPS:
        settings.update(changes)  # cumulative: the row above plus this one change
        hw4.CONFIG = dict(FINAL)
        hw4.CONFIG.update(settings)
        start = time.perf_counter()
        scores = run_config(corpus, questions, ground_truth)
        elapsed = time.perf_counter() - start
        row = {"configuration": label, "change": ", ".join(f"{k}={v}" for k, v in changes.items()),
               "reason": reason, **{k: (round(v, 3) if isinstance(v, float) else v)
                                    for k, v in scores.items()},
               "seconds": round(elapsed, 1)}
        rows.append(row)
        print(f"{label:26} recall {scores['recall']:.1%}  kept {scores['kept']:.1%}  "
              f"rejected {scores['rejected']:.1%}  accuracy {scores['accuracy']:.1%}  "
              f"({elapsed:.0f}s)", flush=True)

    fields = ["configuration", "change", "reason", "parse", "recall", "kept", "rejected", "accuracy",
              "avg_chunks", "points_2_2", "points_2_3", "points_3", "points_4", "table_note", "seconds"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.output}")
    print(f"Your usage: {hw4.USAGE.summary()}")
    print(f"Generator usage: {hw4.GENERATOR_USAGE.summary()}")


if __name__ == "__main__":
    main()
