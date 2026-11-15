
import json
import time
from pathlib import Path

from src.rag_pipeline import initialize_retriever
from src.vector_store import semantic_search
from src.reranker import load_reranker, rerank_documents


BASE_DIR = Path(__file__).resolve().parent.parent

QUESTIONS_FILE = (
    BASE_DIR / "evaluation" / "questions.json"
)

OUTPUT_FILE = (
    BASE_DIR / "evaluation" / "results"
    / "reranking_results.json"
)

TOP_K = 5
CANDIDATE_K = 20


def precision_at_k(ids, relevant_ids, k):
    relevant = set(relevant_ids)

    return sum(
        chunk_id in relevant
        for chunk_id in ids[:k]
    ) / k


def reciprocal_rank(ids, relevant_ids):
    relevant = set(relevant_ids)

    for rank, chunk_id in enumerate(ids, 1):
        if chunk_id in relevant:
            return 1.0 / rank

    return 0.0


def main():

    with QUESTIONS_FILE.open(
        "r", encoding="utf-8"
    ) as f:
        questions = json.load(f)

    embedding_model, collection = initialize_retriever()
    reranker = load_reranker()

    all_results = []

    for index, item in enumerate(questions, 1):

        question = item["question"]
        relevant_ids = set(
            item["relevant_chunk_ids"]
        )

        if not relevant_ids:
            continue

        # Même recherche initiale pour les deux méthodes
        start = time.perf_counter()

        candidates = semantic_search(
            question=question,
            model=embedding_model,
            collection=collection,
            top_k=CANDIDATE_K
        )

        retrieval_time = (
            time.perf_counter() - start
        )

        # Baseline ChromaDB
        baseline = candidates[:TOP_K]

        # Cross-Encoder
        start = time.perf_counter()

        reranked = rerank_documents(
            question=question,
            documents=candidates,
            reranker=reranker,
            top_k=TOP_K
        )

        reranking_time = (
            time.perf_counter() - start
        )

        baseline_ids = [
            r["chunk_id"] for r in baseline
        ]

        reranked_ids = [
            r["chunk_id"] for r in reranked
        ]

        p_before = precision_at_k(
            baseline_ids, relevant_ids, TOP_K
        )

        p_after = precision_at_k(
            reranked_ids, relevant_ids, TOP_K
        )

        rr_before = reciprocal_rank(
            baseline_ids, relevant_ids
        )

        rr_after = reciprocal_rank(
            reranked_ids, relevant_ids
        )

        print(f"\nQUESTION {index} : {question}")

        print(
            f"Precision@5 : "
            f"{p_before:.3f} → {p_after:.3f}"
        )

        print(
            f"RR@5 : "
            f"{rr_before:.3f} → {rr_after:.3f}"
        )

        print(
            f"Temps de reranking : "
            f"{reranking_time:.3f} s"
        )

        all_results.append({
            "question": question,
            "baseline_ids": baseline_ids,
            "reranked_ids": reranked_ids,
            "precision_before": p_before,
            "precision_after": p_after,
            "rr_before": rr_before,
            "rr_after": rr_after,
            "retrieval_seconds": retrieval_time,
            "reranking_seconds": reranking_time
        })

    if not all_results:
        print("Aucune question annotée.")
        return

    print("\n===== COMPARAISON GLOBALE =====")

    for metric in [
        "precision_before",
        "precision_after",
        "rr_before",
        "rr_after",
        "reranking_seconds"
    ]:
        average = sum(
            result[metric]
            for result in all_results
        ) / len(all_results)

        print(f"{metric}: {average:.4f}")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            all_results,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"\nRésultats : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
