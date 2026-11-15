
import json
import time
from pathlib import Path
from datetime import datetime, timezone

from src.rag_pipeline import (
    initialize_retriever,
    run_rag
)


BASE_DIR = Path(__file__).resolve().parent.parent

QUESTIONS_FILE = (
    BASE_DIR / "evaluation" / "questions.json"
)

OUTPUT_DIR = BASE_DIR / "evaluation" / "results"
OUTPUT_FILE = OUTPUT_DIR / "generation_results.json"

PROVIDERS = ("ollama", "openai")
TOP_K = 5


def load_questions():
    with QUESTIONS_FILE.open(
        "r", encoding="utf-8"
    ) as f:
        return json.load(f)


def save_results(results):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )


def evaluate_generation():

    questions = load_questions()

    model, collection = initialize_retriever()

    evaluations = []

    for index, item in enumerate(questions, 1):

        question = item["question"]

        print(f"\n{'=' * 60}")
        print(f"QUESTION {index}/{len(questions)}")
        print(question)
        print("=" * 60)

        result = run_rag(
            question=question,
            embedding_model=model,
            collection=collection,
            providers=PROVIDERS,
            top_k=TOP_K
        )

        if not result:
            print("Aucun résultat.")
            continue

        evaluation = {
            "question": question,
            "top_k": TOP_K,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "retrieval_seconds": result[
                "retrieval_seconds"
            ],
            "sources": [
                {
                    "chunk_id": source.get("chunk_id"),
                    "source": source["metadata"].get("source"),
                    "page": source["metadata"].get("page"),
                    "similarity": source["similarity"],
                    "text": source["text"]
                }
                for source in result["sources"]
            ],
            "responses": result["responses"]
        }

        evaluations.append(evaluation)

        # Sauvegarde après chaque question
        save_results(evaluations)

        for provider, response in result["responses"].items():

            if "error" in response:
                print(
                    f"{provider} : ERREUR - "
                    f"{response['error']}"
                )
                continue

            print(
                f"{provider} : "
                f"{response['generation_seconds']:.2f} s"
            )

    print("\n===== RÉSUMÉ =====")

    for provider in PROVIDERS:

        durations = []

        for item in evaluations:
            response = item["responses"].get(
                provider, {}
            )

            if "generation_seconds" in response:
                durations.append(
                    response["generation_seconds"]
                )

        if durations:
            average = sum(durations) / len(durations)

            print(
                f"{provider} : "
                f"{average:.2f} secondes en moyenne "
                f"sur {len(durations)} réponses"
            )

    print(f"\nRésultats enregistrés : {OUTPUT_FILE}")


if __name__ == "__main__":
    evaluate_generation()
