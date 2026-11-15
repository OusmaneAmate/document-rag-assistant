
from src.rag_pipeline import initialize_retriever
from src.vector_store import semantic_search
from src.reranker import (
    load_reranker,
    rerank_documents
)


def main():

    question = (
        "How does PCA affect clustering performance?"
    )

    embedding_model, collection = initialize_retriever()

    reranker = load_reranker()

    # Première étape : recherche vectorielle
    candidates = semantic_search(
        question=question,
        model=embedding_model,
        collection=collection,
        top_k=20
    )

    # Deuxième étape : reranking
    reranked = rerank_documents(
        question=question,
        documents=candidates,
        reranker=reranker,
        top_k=5
    )

    print("\n===== AVANT RERANKING =====")

    for rank, result in enumerate(candidates[:5], 1):
        print(
            f"{rank}. {result['chunk_id']} | "
            f"Cosine : {result['similarity']:.4f}"
        )

    print("\n===== APRÈS RERANKING =====")

    for rank, result in enumerate(reranked, 1):

        print(
            f"{rank}. {result['chunk_id']} | "
            f"Rang initial : {result['original_rank']} | "
            f"Rerank score : {result['rerank_score']:.4f}"
        )

        print(
            "Source :",
            result["metadata"].get("source")
        )

        print(
            "Texte :",
            result["text"][:350]
        )

        print("-" * 50)


if __name__ == "__main__":
    main()
