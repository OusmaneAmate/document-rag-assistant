
import json
from pathlib import Path
from statistics import mean

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "documents.json"
OUTPUT_DIR = BASE_DIR / "data" / "processed"


def load_documents(path=INPUT_FILE):
    """Charge les pages extraites à l'étape 2."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_fixed(text, chunk_size=1000, chunk_overlap=200):
    """Découpage fixe par caractères."""
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Paramètres de chunking invalides.")

    step = chunk_size - chunk_overlap
    chunks = []

    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size]

        if chunk.strip():
            chunks.append(chunk)

        if start + chunk_size >= len(text):
            break

    return chunks


def split_recursive(text, chunk_size=1000, chunk_overlap=200):
    """Découpage récursif avec LangChain."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    return splitter.split_text(text)


def create_chunks(
    documents,
    strategy="recursive",
    chunk_size=1000,
    chunk_overlap=200
):
    """Découpe chaque page tout en conservant les métadonnées."""
    if strategy not in ("fixed", "recursive"):
        raise ValueError("Stratégie inconnue.")

    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Paramètres de chunking invalides.")

    all_chunks = []

    for document in documents:
        text = document["text"]

        if not text.strip():
            continue

        if strategy == "fixed":
            pieces = split_fixed(
                text, chunk_size, chunk_overlap
            )
        else:
            pieces = split_recursive(
                text, chunk_size, chunk_overlap
            )

        for index, piece in enumerate(pieces, start=1):
            if not piece.strip():
                continue

            all_chunks.append({
                "chunk_id": (
                    f"chunk_{len(all_chunks) + 1:06d}"
                ),
                "text": piece,
                "metadata": {
                    **document["metadata"],
                    "page_id": document["page_id"],
                    "chunk_index": index,
                    "strategy": strategy,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap
                }
            })

    return all_chunks


def chunk_statistics(chunks):
    """Calcule les statistiques du découpage."""
    lengths = [len(c["text"]) for c in chunks]

    return {
        "number_of_chunks": len(chunks),
        "average_length": round(mean(lengths), 2)
        if lengths else 0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0
    }


def save_chunks(chunks, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Fichier enregistré : {output_path}")


def main():
    documents = load_documents()

    configurations = [
        ("fixed", 1000, 200),
        ("recursive", 1000, 200),
        ("recursive", 500, 100)
    ]

    for strategy, size, overlap in configurations:
        chunks = create_chunks(
            documents,
            strategy=strategy,
            chunk_size=size,
            chunk_overlap=overlap
        )

        stats = chunk_statistics(chunks)

        print(f"\nStratégie : {strategy}")
        print(f"Chunk size : {size}")
        print(f"Overlap : {overlap}")
        print(f"Statistiques : {stats}")

        filename = (
            f"chunks_{strategy}_{size}_{overlap}.json"
        )

        save_chunks(chunks, filename)

        if chunks:
            print("\nExemple de chunk :")
            print(chunks[0]["text"][:400])


if __name__ == "__main__":
    main()
