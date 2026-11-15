
import csv
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "evaluation"
    / "results"
    / "generation_results.json"
)

OUTPUT_FILE = (
    BASE_DIR
    / "evaluation"
    / "results"
    / "generation_annotations.csv"
)


def main():

    with INPUT_FILE.open(
        "r", encoding="utf-8"
    ) as f:
        evaluations = json.load(f)

    rows = []

    for index, item in enumerate(evaluations, 1):

        for provider, response in item["responses"].items():

            if "error" in response:
                continue

            rows.append({
                "question_id": f"Q{index:03d}",
                "question": item["question"],
                "provider": provider,
                "model": response["model"],
                "answer": response["answer"],
                "faithfulness": "",
                "answer_relevance": "",
                "citation_accuracy": "",
                "comments": ""
            })

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} réponses exportées.")
    print(f"Fichier : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
