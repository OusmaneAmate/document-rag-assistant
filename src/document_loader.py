
from pathlib import Path
from typing import Any
import json
import logging
import re

import pymupdf  
import pymupdf4llm


# --------------------------------------------------
# 1. Configuration des chemins
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "documents.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 2. Nettoyage du texte
# --------------------------------------------------

def clean_text(text: str) -> str:
    """
    Nettoie légèrement le texte extrait.

    - Supprime les caractères nuls
    - Normalise certains espaces
    - Réduit les espaces répétitifs
    - Limite les lignes vides consécutives
    """

    text = text.replace("\x00", "")
    text = text.replace("\u00a0", " ")

    # Réduit les espaces horizontaux
    text = re.sub(r"[ \t]+", " ", text)

    # Supprime les espaces en fin de ligne
    text = re.sub(r" *\n *", "\n", text)

    # Limite les lignes vides
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# --------------------------------------------------
# 3. Chargement d'un fichier PDF
# --------------------------------------------------


def load_pdf(pdf_path: Path) -> list[dict[str, Any]]:
    """
    Charge un PDF avec PyMuPDF4LLM.

    Extrait le texte au format Markdown en tenant
    compte de la mise en page du document.
    """

    documents = []

    logger.info("Chargement : %s", pdf_path.name)

    try:
        with pymupdf.open(pdf_path) as pdf:
            if pdf.needs_pass:
                logger.warning(
                    "PDF protégé : %s",
                    pdf_path.name
                )
                return []

            total_pages = len(pdf)
            metadata = pdf.metadata or {}
            title = metadata.get("title") or pdf_path.stem

            # Extraction page par page au format Markdown
            pages = pymupdf4llm.to_markdown(
                pdf,
                page_chunks=True,
                show_progress=False
            )

            for index, page_data in enumerate(pages):
                text = clean_text(page_data.get("text", ""))

                if not text:
                    logger.warning(
                        "Page vide : %s, page %s",
                        pdf_path.name,
                        index + 1
                    )

                documents.append({
                    "page_id": (
                        f"{pdf_path.stem}_page_{index + 1}"
                    ),
                    "text": text,
                    "metadata": {
                        "source": pdf_path.name,
                        "title": title,
                        "page": index + 1,
                        "total_pages": total_pages,
                        "file_path": str(
                            pdf_path.relative_to(BASE_DIR)
                        )
                    }
                })

            logger.info(
                "%s : %s pages traitées",
                pdf_path.name,
                len(documents)
            )

    except Exception as error:
        logger.error(
            "Erreur de lecture %s : %s",
            pdf_path.name,
            error
        )

    return documents


# --------------------------------------------------
# 4. Chargement du corpus documentaire
# --------------------------------------------------

def load_corpus(
    directory: Path = DOCUMENTS_DIR
) -> list[dict[str, Any]]:
    """
    Parcourt un dossier et charge tous les PDF.
    """

    if not directory.exists():
        raise FileNotFoundError(
            f"Dossier introuvable : {directory}"
        )

    pdf_files = sorted(
        path for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".pdf"
    )

    if not pdf_files:
        logger.warning(
            "Aucun fichier PDF trouvé dans %s",
            directory
        )
        return []

    logger.info(
        "%s fichiers PDF trouvés",
        len(pdf_files)
    )

    corpus = []

    for pdf_path in pdf_files:
        pages = load_pdf(pdf_path)
        corpus.extend(pages)

    return corpus


# --------------------------------------------------
# 5. Statistiques du corpus
# --------------------------------------------------

def corpus_statistics(
    corpus: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Calcule des statistiques sur les pages extraites.
    """

    total_pages = len(corpus)

    pages_with_text = sum(
        bool(doc["text"]) for doc in corpus
    )

    empty_pages = total_pages - pages_with_text

    total_words = sum(
        len(doc["text"].split())
        for doc in corpus
    )

    total_characters = sum(
        len(doc["text"])
        for doc in corpus
    )

    unique_sources = {
        doc["metadata"]["file_path"]
        for doc in corpus
    }

    return {
        "total_documents": len(unique_sources),
        "total_pages": total_pages,
        "pages_with_text": pages_with_text,
        "empty_pages": empty_pages,
        "total_words": total_words,
        "total_characters": total_characters,
        "average_words_per_page": round(
            total_words / total_pages, 2
        ) if total_pages else 0
    }


# --------------------------------------------------
# 6. Sauvegarde au format JSON
# --------------------------------------------------

def save_corpus(
    corpus: list[dict[str, Any]],
    output_path: Path = OUTPUT_FILE
) -> None:
    """
    Sauvegarde les pages extraites au format JSON.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            corpus,
            file,
            ensure_ascii=False,
            indent=2
        )

    logger.info(
        "Corpus enregistré : %s",
        output_path
    )


# --------------------------------------------------
# 7. Fonction principale
# --------------------------------------------------

def main():

    print("\n===== CHARGEMENT DU CORPUS =====\n")

    corpus = load_corpus()

    if not corpus:
        print("Aucune page n'a été chargée.")
        return

    stats = corpus_statistics(corpus)

    print("\n===== STATISTIQUES =====\n")

    for key, value in stats.items():
        print(f"{key}: {value}")

    save_corpus(corpus)

    print("\n===== EXEMPLE DE TEXTE EXTRAIT =====\n")

    first_doc = next(
        (doc for doc in corpus if doc["text"]),
        None
    )

    if first_doc:
        print("Source :", first_doc["metadata"]["source"])
        print("Page :", first_doc["metadata"]["page"])
        print("\nTexte :\n")
        print(first_doc["text"][:1000])
    else:
        print("Aucun texte exploitable trouvé.")

    print("\n===== FIN DU TRAITEMENT =====\n")


if __name__ == "__main__":
    main()
