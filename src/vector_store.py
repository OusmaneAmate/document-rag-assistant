
import json
from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
CHROMA_DIR = BASE_DIR / "chroma_db"

EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"
CHUNKS_FILE = DATA_DIR / "indexed_chunks.json"

COLLECTION_NAME = "scientific_documents"

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# --------------------------------------------------
# 2. Chargement des données existantes
# --------------------------------------------------

def load_data():
    """Charge les embeddings et les chunks associés."""

    embeddings = np.load(
        EMBEDDINGS_FILE,
        allow_pickle=False
    )

    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    if embeddings.ndim != 2:
        raise ValueError("Embeddings : tableau 2D attendu.")

    if len(embeddings) != len(chunks):
        raise ValueError(
            "Le nombre de vecteurs ne correspond "
            "pas au nombre de chunks."
        )

    if embeddings.shape[1] != 384:
        raise ValueError(
            "Dimensions inattendues des embeddings."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings non finis détectés.")

    print(f"Chunks chargés : {len(chunks)}")
    print(f"Dimensions : {embeddings.shape}")

    return chunks, embeddings


# --------------------------------------------------
# 3. Connexion à ChromaDB
# --------------------------------------------------

def get_client():
    """
    Crée un client ChromaDB persistant.

    Les données seront sauvegardées dans chroma_db/.
    """

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    return chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )


# --------------------------------------------------
# 4. Création et indexation de la collection
# --------------------------------------------------

def create_vector_store(
    chunks,
    embeddings,
    batch_size=100
):
    """
    Crée une collection et insère les chunks.

    Cette fonction reconstruit la collection dédiée
    au projet si elle existe déjà.
    """

    client = get_client()

    # Reconstruction propre de notre collection
    if COLLECTION_NAME in [
        collection.name
        for collection in client.list_collections()
    ]:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        configuration={
            "hnsw": {
                "space": "cosine"
            }
        },
        metadata={
            "description": "Corpus scientifique RAG",
            "embedding_model": MODEL_NAME
        }
    )

    # Insertion des données par lots
    for start in range(0, len(chunks), batch_size):

        end = min(start + batch_size, len(chunks))

        batch_chunks = chunks[start:end]
        batch_embeddings = embeddings[start:end]

        ids = [
            chunk["chunk_id"]
            for chunk in batch_chunks
        ]

        texts = [
            chunk["text"]
            for chunk in batch_chunks
        ]

        metadatas = [
            {
                key: value
                for key, value in chunk["metadata"].items()
                if value is not None
                and isinstance(value, (str, int, float, bool))
            }
            for chunk in batch_chunks
        ]

        collection.upsert(
            ids=ids,
            embeddings=batch_embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )

        print(
            f"Indexation : {end}/{len(chunks)} chunks"
        )

    print("\nCollection créée avec succès.")
    print("Nombre d'éléments :", collection.count())

    return collection


# --------------------------------------------------
# 5. Recherche sémantique
# --------------------------------------------------

def semantic_search(
    question,
    model,
    collection,
    top_k=5,
    source=None
):
    """
    Recherche les chunks les plus proches.

    source : permet de filtrer par nom de PDF.
    """

    if top_k < 1:
        raise ValueError("top_k doit être positif.")

    if collection.count() == 0:
        return []

    # Embedding de la question
    query_embedding = model.encode_query(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Filtrage facultatif par source
    where_filter = (
        {"source": source}
        if source is not None
        else None
    )

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(top_k, collection.count()),
        where=where_filter,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    formatted_results = []

    for chunk_id, text, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
  ):
        formatted_results.append({
          "chunk_id": chunk_id,
          "text": text,
          "metadata": metadata,
          "distance": float(distance),
          "similarity": 1.0 - float(distance)
        })

    return formatted_results


# --------------------------------------------------
# 6. Programme principal
# --------------------------------------------------

def main():

    print("\n===== INDEXATION CHROMADB =====\n")

    chunks, embeddings = load_data()

    collection = create_vector_store(
        chunks,
        embeddings
    )

    print("\n===== RECHERCHE SEMANTIQUE =====\n")

    model = SentenceTransformer(MODEL_NAME)

    question = (
        "What is the impact of dimensionality "
        "reduction on clustering?"
    )

    results = semantic_search(
        question=question,
        model=model,
        collection=collection,
        top_k=5
    )

    print("Question :", question)

    for i, result in enumerate(results, start=1):

        metadata = result["metadata"]

        print(f"\n===== RESULTAT {i} =====")
        print(
            f"Similarité : {result['similarity']:.4f}"
        )
        print(
            f"Distance : {result['distance']:.4f}"
        )
        print("Source :", metadata.get("source"))
        print("Page :", metadata.get("page"))
        print("Texte :", result["text"][:600])


if __name__ == "__main__":
    main()
