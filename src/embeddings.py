
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = DATA_DIR / "chunks_recursive_1000_200.json"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"
CHUNKS_FILE = DATA_DIR / "indexed_chunks.json"

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

MIN_CHARS = 50


def load_chunks():
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Filtrage des chunks très courts
    valid_chunks = [
        chunk for chunk in chunks
        if len(chunk["text"].strip()) >= MIN_CHARS
    ]

    print(f"Chunks initiaux : {len(chunks)}")
    print(f"Chunks conservés : {len(valid_chunks)}")
    print(f"Chunks exclus : {len(chunks) - len(valid_chunks)}")

    return valid_chunks


def generate_embeddings(model, chunks):
    texts = [chunk["text"] for chunk in chunks]

    print("\nGénération des embeddings...")

    embeddings = model.encode_document(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return np.asarray(embeddings, dtype=np.float32)


def save_results(chunks, embeddings):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    np.save(EMBEDDINGS_FILE, embeddings)

    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\nEmbeddings et chunks enregistrés.")


def semantic_search(
    query,
    model,
    embeddings,
    chunks,
    top_k=5
):
    # Vectorisation de la question
    query_embedding = model.encode_query(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )[0]

    # Produit scalaire = similarité cosinus
    # puisque les vecteurs sont normalisés
    similarities = embeddings @ query_embedding

    top_k = min(top_k, len(chunks))

    best_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for idx in best_indices:
        chunk = chunks[int(idx)]

        results.append({
            "score": float(similarities[idx]),
            "text": chunk["text"],
            "metadata": chunk["metadata"]
        })

    return results


def main():
    chunks = load_chunks()

    if not chunks:
        raise ValueError("Aucun chunk exploitable.")

    print("\nChargement du modèle...")
    model = SentenceTransformer(MODEL_NAME)

    embeddings = generate_embeddings(model, chunks)

    print("\nDimensions :", embeddings.shape)

    save_results(chunks, embeddings)

    query = (
        "What is the impact of dimensionality "
        "reduction on clustering?"
    )

    print("\nQUESTION :", query)

    results = semantic_search(
        query,
        model,
        embeddings,
        chunks,
        top_k=5
    )

    for rank, result in enumerate(results, start=1):
        print(f"\n===== RESULTAT {rank} =====")
        print(f"Similarité : {result['score']:.4f}")
        print(f"Source : {result['metadata']['source']}")
        print(f"Page : {result['metadata']['page']}")
        print(f"Texte : {result['text'][:600]}")


if __name__ == "__main__":
    main()
