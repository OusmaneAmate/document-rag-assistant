# Document RAG Assistant

**An intelligent document question-answering system powered by Retrieval-Augmented Generation (RAG), semantic search, and Cross-Encoder reranking.**

This project implements an end-to-end RAG pipeline that enables users to ask natural-language questions about a collection of PDF documents and receive context-aware answers with source citations.

It supports both **local LLM inference using Ollama** and **cloud-based generation using the OpenAI API**, allowing comparisons of answer quality, retrieval performance, and inference latency.

## 1. Project Overview

Large collections of documents often contain valuable information that is difficult to retrieve efficiently using traditional keyword-based search.

This project addresses that challenge by combining:

- PDF document extraction and preprocessing
- Text chunking and multilingual embeddings
- Semantic retrieval with a persistent vector database
- Cross-Encoder reranking
- LLM-based answer generation
- Source attribution
- Retrieval and generation evaluation

The initial implementation focuses on scientific documents related to dimensionality reduction and clustering.

The architecture is designed to be reusable with other text-based PDF collections.

## 2. System Architecture

The RAG pipeline consists of two main phases.

### Document Indexing

```text
PDF Documents
     |
     v
Text Extraction (PyMuPDF4LLM)
     |
     v
Text Cleaning and Chunking
     |
     v
Embedding Generation
(Sentence Transformers)
     |
     v
Vector Database (ChromaDB)
```

### Retrieval and Answer Generation

```text
User Question
     |
     v
Query Embedding
     |
     v
Semantic Search (ChromaDB)
     |
     v
Top-20 Candidate Passages
     |
     v
Cross-Encoder Reranking (Optional)
     |
     v
Top-5 Relevant Passages
     |
     +-----------------------+
     |                       |
     v                       v
Ollama / Llama 3.2      OpenAI / GPT-5 mini
     |                       |
     v                       v
Generated Answer        Generated Answer
     |                       |
     +-----------+-----------+
                 |
                 v
        Answer Comparison
        and Source Citations
```

The reranking stage is optional, allowing direct comparison between standard semantic retrieval and reranked retrieval.

## 3. Key Features

### Document Processing

- Automatic loading of multiple PDF documents
- Markdown-based text extraction with PyMuPDF4LLM
- Preservation of source filenames and page numbers
- Text normalization and preprocessing
- Fixed-size and recursive text chunking
- Filtering of very short text chunks

### Semantic Retrieval

- Multilingual sentence embeddings
- Persistent vector storage with ChromaDB
- Cosine similarity search
- Configurable Top-k retrieval
- Retrieval of source metadata

### Cross-Encoder Reranking

- Two-stage retrieval architecture
- Initial candidate selection using ChromaDB
- Relevance scoring using a multilingual Cross-Encoder
- Selection of the highest-ranked passages
- Comparison of retrieval performance before and after reranking

### LLM Integration

- Local inference using Llama 3.2 through Ollama
- Cloud inference using GPT-5 mini through the OpenAI API
- Shared retrieved context for model comparison
- Source-aware prompt construction
- Instructions designed to reduce unsupported answers

### Interactive Web Application

A Streamlit interface allows users to:

- Ask questions in natural language
- Select Ollama or OpenAI
- Compare both models
- Enable or disable reranking
- Configure retrieval parameters
- Examine retrieved passages and source references
- Inspect generation and retrieval latency

## 4. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| PDF Processing | PyMuPDF, PyMuPDF4LLM |
| Text Chunking | LangChain Text Splitters |
| Embeddings | Sentence Transformers |
| Vector Database | ChromaDB |
| Reranking | Cross-Encoder |
| Local LLM | Llama 3.2 (3B) |
| Local Inference | Ollama |
| Cloud LLM | GPT-5 mini |
| API Integration | OpenAI Python SDK |
| Web Interface | Streamlit |
| Numerical Computing | NumPy |
| Evaluation | Python, Precision@k, MRR |

### Models

**Embedding model**

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

- Multilingual semantic representations
- 384-dimensional embeddings
- Used for document indexing and query encoding

**Reranking model**

`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`

- Multilingual Cross-Encoder
- Scores question–passage pairs
- Reorders retrieved candidates based on relevance

**Generation models**

- `llama3.2:3b` — local inference
- `gpt-5-mini` — API-based inference

## 5. Project Structure

```text
document-rag-assistant/
|
├── src/
│   ├── __init__.py
│   ├── document_loader.py
│   ├── text_splitter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── reranker.py
│   └── rag_pipeline.py
|
├── evaluation/
│   ├── questions.json
│   ├── evaluate_retrieval.py
│   ├── evaluate_generation.py
│   ├── evaluate_reranking.py
│   └── inspect_reranking.py
|
├── tests/
│   ├── test_document_loader.py
│   └── test_reranker.py
|
├── data/
│   ├── documents/
│   └── processed/
|
├── chroma_db/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

The source documents, processed data, vector database, and generated evaluation results are excluded from version control.

## 6. Installation and Setup

### Prerequisites

- Python 3.11
- Conda or another Python environment manager
- Git
- Ollama for local LLM inference
- OpenAI API key for cloud inference (optional)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/document-rag-assistant.git
cd document-rag-assistant
```

Replace `YOUR_USERNAME` with your GitHub username.

### Step 2 — Create the Python Environment

```bash
conda create -n rag_assistant python=3.11 -y
conda activate rag_assistant
```

### Step 3 — Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4 — Configure Environment Variables

Copy `.env.example` to `.env` and configure:

```dotenv
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-5-mini
OLLAMA_MODEL=llama3.2:3b
```

An OpenAI API key is only required for OpenAI inference.

Never commit your `.env` file or API keys.

### Step 5 — Install the Local Model

Install Ollama from:

https://ollama.com/download

Download Llama 3.2:

```bash
ollama pull llama3.2:3b
```

Verify the installation:

```bash
ollama list
```

### Step 6 — Add PDF Documents

Create the following directories if needed:

```text
data/documents/
data/processed/
```

Place your PDF files inside `data/documents/`.

The initial pipeline works best with text-based PDF documents. Scanned PDFs may require OCR.

### Step 7 — Build the Document Index

Execute the processing pipeline in the following order:

```bash
python -m src.document_loader
python -m src.text_splitter
python -m src.embeddings
python -m src.vector_store
```

These commands:

1. Extract text and metadata from PDF documents.
2. Split the text into chunks.
3. Generate document embeddings.
4. Store the vectors and metadata in ChromaDB.

### Step 8 — Launch the Application

```bash
python -m streamlit run app.py
```

The application will generally be available at:

http://localhost:8501

## 7. Usage Example

**Question**

> What is the impact of dimensionality reduction on clustering?

**Example of a grounded answer**

Dimensionality reduction can improve clustering efficiency by reducing memory requirements and the cost of distance calculations. It may also help mitigate the influence of irrelevant features and noise. However, methods such as PCA and MDS do not guarantee preservation of the original cluster structure.

The application displays the retrieved passages, filenames, page numbers, and similarity scores used to generate the answer.

Actual answers depend on the indexed documents and the selected LLM.

## 8. Retrieval Evaluation

The retrieval component was initially evaluated using five questions related to dimensionality reduction and clustering.

The evaluation uses manually identified relevant passages and ranking-based metrics.

### Initial Retrieval Performance

| Metric | Result |
|---|---:|
| Precision@3 | 0.5333 |
| Precision@5 | 0.4000 |
| Precision@10 | 0.4000 |
| MRR@10 | 0.8667 |

### Impact of Cross-Encoder Reranking

The following experiment compared ChromaDB Top-5 retrieval with a pipeline retrieving 20 candidates and reranking them to select the final Top-5.

| Metric | Baseline RAG | RAG with Reranking |
|---|---:|---:|
| Precision@5 | 0.4000 | 0.6000 |
| MRR@5 | 0.8667 | 0.9000 |

The observed Precision@5 increased by **20 percentage points** on this initial benchmark.

The reranking stage added approximately 3.31 seconds per question on the test machine.

**Evaluation limitation:** The relevance judgments were initially derived from top-ranked retrieval results rather than exhaustive independent annotation of the entire corpus. Therefore, these scores should be considered preliminary and may be affected by incomplete relevance judgments.

## 9. LLM Performance Comparison

Five questions were evaluated with Llama 3.2 and GPT-5 mini using the same retrieval configuration.

### Average Generation Latency

| Model | Average Latency |
|---|---:|
| Llama 3.2 (Ollama) | 60.29 s |
| GPT-5 mini (OpenAI API) | 8.23 s |

In this experiment, GPT-5 mini generated responses approximately **7.3 times faster** than the local Llama 3.2 configuration.

These results are hardware- and environment-dependent and should not be interpreted as universal model benchmarks.

### Qualitative Findings

Both models generally produced relevant answers grounded in the retrieved scientific passages.

However, several limitations were observed:

- Occasional confusion between supervised classification and clustering results
- Incorrect or unsupported source attribution
- Overgeneralization of findings from retrieved passages
- Variation in response detail and structure

Improving retrieval quality and citation reliability remains an important objective.

## 10. Current Limitations

- PDF extraction may produce formatting artifacts for complex tables and mathematical expressions.
- Scanned PDF support is limited without OCR.
- The relevance benchmark currently contains only five annotated questions.
- Some relevance judgments are incomplete.
- LLM-generated citations are not yet automatically validated.
- The system does not yet provide full conversational memory across follow-up questions.
- Local generation speed depends on available computing resources.
- API-based generation requires an external provider and may incur usage costs.

## 11. Future Improvements

Planned improvements include:

- Larger and more independently annotated evaluation datasets
- Systematic comparison of chunking strategies
- Evaluation of alternative embedding and reranking models
- Hybrid retrieval combining lexical and semantic search
- Automated answer faithfulness evaluation
- Citation validation
- Conversational question reformulation
- Query expansion
- Docker containerization
- Deployment of an online demonstration
- Improved support for complex PDF layouts and tables

## 12. Security and Data Privacy

- API credentials are stored in local environment variables.
- Source PDF documents are excluded from the repository.
- Generated embeddings and ChromaDB data are not committed.
- Local LLM inference can be performed without sending document excerpts to a cloud LLM provider.
- When cloud inference is enabled, retrieved context is transmitted to the selected API provider.

Users should only process documents they are authorized to use and should review the privacy policies of any external model providers.

## 13. Project Status

**Status: Active Development**

The current implementation includes document processing, semantic search, vector storage, optional Cross-Encoder reranking, local and API-based generation, a Streamlit interface, and initial evaluation workflows.

Further evaluation and optimization are ongoing.