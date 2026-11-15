
import streamlit as st
from src.reranker import load_reranker

from src.rag_pipeline import (
    initialize_retriever,
    run_rag
)


# ----------------------------------------
# 1. Configuration
# ----------------------------------------

st.set_page_config(
    page_title="Assistant RAG",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Document Assistant")

st.caption(
    "Query your scientific documents "
    "with Ollama and OpenAI."
)


# ----------------------------------------
# 2. Chargement des ressources
# ----------------------------------------

@st.cache_resource
def load_resources():
    return initialize_retriever()

@st.cache_resource
def load_reranker_resource():
    return load_reranker()

try:
    embedding_model, collection = load_resources()

except Exception as error:
    st.error(f"Erreur d'initialisation : {error}")
    st.stop()


# ----------------------------------------
# 3. Barre latérale
# ----------------------------------------

with st.sidebar:

    st.header("Configuration")

    mode = st.selectbox(
        "Generation model",
        [
            "Compare the two",
            "Ollama only",
            "OpenAI only"
        ]
    )

    top_k = st.slider(
        "Number of passes (Top-k)",
        min_value=1,
        max_value=10,
        value=5
    )
    
    use_reranking = st.checkbox(
        "Enable reranking",
        value=True
    )

    candidate_k = st.slider(
       "Number of candidates in ChromaDB",
       min_value=10,
       max_value=50,
       value=20,
       step=5,
       disabled=not use_reranking
  )

    st.divider()

    st.metric(
        "Indexed chunks",
        collection.count()
    )

    st.caption(
        "Embedding : multilingual-MiniLM-L12-v2"
    )

    if st.button(
        "Clear conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()


# ----------------------------------------
# 4. Historique de conversation
# ----------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


def display_answer(response, provider):

    if "error" in response:
        st.error(response["error"])
        return

    st.markdown(response["answer"])

    st.caption(
        f"Model : {response['model']} | "
        f"Generation : "
        f"{response['generation_seconds']:.2f} s"
    )

    if "usage" in response:
        st.caption(
            f"Tokens : {response['usage']}"
        )


def display_sources(sources):

    with st.expander(
        f"See the {len(sources)} retrieved passages"
    ):

        for i, source in enumerate(sources, 1):

            metadata = source["metadata"]

            st.markdown(
                f"**[{i}] {metadata.get('source')} "
                f"— Page {metadata.get('page')}**"
            )

            st.caption(
                f"Similarité : "
                f"{source['similarity']:.4f}"
            )
            
            if "rerank_score" in source:
               st.caption(
                  f"Rang initial : "
                  f"{source['original_rank']} | "
                  f"Score reranking : "
                  f"{source['rerank_score']:.4f}"
                )

            st.text(source["text"])
            st.divider()


def display_result(result, mode_used):

    st.caption(
        "Recherche documentaire : "
        f"{result['retrieval_seconds']:.3f} s"
    )
    if result.get("use_reranking"):
        st.caption(
           "Reranking : "
           f"{result['reranking_seconds']:.3f} s"
        )

    responses = result["responses"]

    if mode_used == "Compare the two":

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🖥️ Ollama")
            display_answer(
                responses.get("ollama", {
                    "error": "Réponse indisponible."
                }),
                "ollama"
            )

        with col2:
            st.subheader("☁️ OpenAI")
            display_answer(
                responses.get("openai", {
                    "error": "Réponse indisponible."
                }),
                "openai"
            )

    else:
        provider = (
            "ollama"
            if mode_used == "Ollama only"
            else "openai"
        )

        display_answer(
            responses.get(provider, {
                "error": "Réponse indisponible."
            }),
            provider
        )

    display_sources(result["sources"])


# Afficher les messages précédents
for message in st.session_state.messages:

    with st.chat_message("user"):
        st.write(message["question"])

    with st.chat_message("assistant"):
        display_result(
            message["result"],
            message["mode"]
        )


# ----------------------------------------
# 5. Nouvelle question
# ----------------------------------------
reranker_model = (
    load_reranker_resource()
    if use_reranking
    else None
)

question = st.chat_input(
    "Ask a question about your documents......"
)

if question:

    with st.chat_message("user"):
        st.write(question)

    if mode == "Compare the two":
        providers = ("ollama", "openai")

    elif mode == "Ollama only":
        providers = ("ollama",)

    else:
        providers = ("openai",)

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching and generating in progress..."
        ):

            result = run_rag(
                question=question,
                embedding_model=embedding_model,
                collection=collection,
                providers=providers,
                top_k=top_k,
                use_reranking=use_reranking,
                reranker=reranker_model,
                candidate_k=candidate_k
            )

        if result:
            display_result(result, mode)

            st.session_state.messages.append({
                "question": question,
                "mode": mode,
                "result": result
            })

        else:
            st.warning(
                "Aucun passage pertinent disponible."
            )
