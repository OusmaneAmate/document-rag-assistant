
from src.rag_pipeline import initialize_retriever
from src.vector_store import semantic_search



def main():

    model, collection = initialize_retriever()

    question = input(
        "\nEntrez votre question : "
    ).strip()

    if not question:
        print("Aucune question saisie.")
        return

    results = semantic_search(
        question=question,
        model=model,
        collection=collection,
        top_k=10
    )

    for i, result in enumerate(results, 1):

        print(f"\n{'=' * 60}")
        print(f"RÉSULTAT {i}")
        print(f"{'=' * 60}")

        print("Chunk ID :", result["chunk_id"])
        print("Score :", round(result["similarity"], 4))
        print("Source :", result["metadata"]["source"])
        print("Page :", result["metadata"]["page"])
        print("\nTexte :\n")
        print(result["text"])


if __name__ == "__main__":
    main()
