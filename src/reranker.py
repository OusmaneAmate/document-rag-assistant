
import numpy as np
from sentence_transformers import CrossEncoder


RERANKER_MODEL = (
    "cross-encoder/"
    "mmarco-mMiniLMv2-L12-H384-v1"
)


def load_reranker():
    """
    Charge le Cross-Encoder multilingue.
    Le modèle est téléchargé au premier lancement.
    """
    print("Chargement du modèle de reranking...")

    model = CrossEncoder(
        RERANKER_MODEL,
        max_length=512
    )

    return model


def rerank_documents(
    question,
    documents,
    reranker,
    top_k=5,
    batch_size=8
):
    """
    Reclasse les documents selon leur pertinence
    pour la question.

    Paramètres
    ----------
    question : str
        Question de l'utilisateur.

    documents : list[dict]
        Résultats du retriever ChromaDB.

    reranker : CrossEncoder
        Modèle de reranking préchargé.

    top_k : int
        Nombre final de documents à conserver.

    Retour
    ------
    list[dict]
        Documents reclassés avec leurs scores.
    """

    if top_k < 1:
        raise ValueError("top_k doit être >= 1")

    if not documents:
        return []

    # Création des paires question-passage
    pairs = [
        (question, document["text"])
        for document in documents
    ]

    # Calcul des scores de reranking
    scores = reranker.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
        apply_softmax=False
    )

    scores = np.asarray(scores).reshape(-1)

    if len(scores) != len(documents):
        raise ValueError(
            "Nombre de scores incorrect."
        )

    # Conservation des résultats et des métadonnées
    reranked = []

    for index, (document, score) in enumerate(
        zip(documents, scores),
        start=1
    ):
        result = document.copy()

        result["original_rank"] = index
        result["rerank_score"] = float(score)

        reranked.append(result)

    # Tri décroissant selon le Cross-Encoder
    reranked.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    # Conservation des meilleurs passages
    return reranked[:top_k]
