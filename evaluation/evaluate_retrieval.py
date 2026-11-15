
import json
from pathlib import Path

from src.rag_pipeline import initialize_retriever
from src.vector_store import semantic_search


BASE_DIR = Path(__file__).resolve().parent.parent

QUESTIONS_FILE = (
    BASE_DIR / "evaluation" / "questions.json"
)

K_VALUES = [3, 5, 10]


def precision_at_k(retrieved_ids, relevant_ids, k):
    """Proportion de chunks pertinents parmi les k premiers."""

    relevant = set(relevant_ids)

    hits = sum(
        chunk_id in relevant
        for chunk_id in retrieved_ids[:k]
    )

    return hits / k


def reciprocal_rank(retrieved_ids, relevant_ids):
    """Inverse du rang du premier chunk pertinent."""

    relevant = set(relevant_ids)

    for rank, chunk_id in enumerate(retrieved_ids, 1):
        if chunk_id in relevant:
            return 1.0 / rank

    return 0.0


def recall_at_k(retrieved_ids, relevant_ids, k):
    """
    Rappel par rapport aux chunks annotés.

    Ce n'est un Recall@k complet que lorsque
    les annotations sont exhaustives.
    """

    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    retrieved = set(retrieved_ids[:k])

    hits = len(retrieved.intersection(relevant))

    return hits / len(relevant)


def main():

    with QUESTIONS_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:
        questions = json.load(file)

    model, collection = initialize_retriever()

    all_results = []

    for item in questions:

        question = item["question"]

        relevant_ids = item.get(
            "relevant_chunk_ids", []
        )

        # Ne pas évaluer les questions non annotées
        if not relevant_ids:
            print(
                f"\nQuestion non annotée : {question}"
            )
            continue

        print(f"\nQUESTION : {question}")

        results = semantic_search(
            question=question,
            model=model,
            collection=collection,
            top_k=max(K_VALUES)
        )

        retrieved_ids = [
            result["chunk_id"]
            for result in results
        ]

        metrics = {}

        for k in K_VALUES:

            precision = precision_at_k(
                retrieved_ids,
                relevant_ids,
                k
            )

            recall = recall_at_k(
                retrieved_ids,
                relevant_ids,
                k
            )

            metrics[f"P@{k}"] = precision
            metrics[f"AnnotatedRecall@{k}"] = recall

            print(
                f"P@{k}: {precision:.4f} | "
                f"AnnotatedRecall@{k}: {recall:.4f}"
            )

        rr = reciprocal_rank(
            retrieved_ids,
            relevant_ids
        )

        metrics["RR"] = rr

        print(f"Reciprocal Rank : {rr:.4f}")

        all_results.append(metrics)

    if not all_results:
        print("\nAucune question annotée.")
        return

    print("\n===== MOYENNES =====")

    for metric in all_results[0]:

        average = sum(
            item[metric] for item in all_results
        ) / len(all_results)

        label = "MRR" if metric == "RR" else metric

        print(f"{label}: {average:.4f}")

    print(
        f"\nQuestions évaluées : "
        f"{len(all_results)}/{len(questions)}"
    )


if __name__ == "__main__":
    main()
