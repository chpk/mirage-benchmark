# MiRAGE: Multimodal Multihop RAG Evaluation Dataset Generator

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License">
  <img src="https://img.shields.io/pypi/v/mirage-bench.svg" alt="PyPI">
</p>

**MiRAGE** is a multi-agent framework for generating high-quality, multimodal, multihop question-answer datasets for evaluating Retrieval-Augmented Generation (RAG) systems. It automatically extracts domain expertise, builds complete context through iterative retrieval, and generates verified QA pairs from technical documents.

<p align="center">
  <img src="assets/mirage_framework.png" alt="MiRAGE Framework Architecture" width="100%">
</p>

## Key Features

- **Multi-hop Context Completion**: Iteratively expands incomplete chunks with relevant context across documents
- **Domain and Expert Role Detection**: Automatic domain identification using BERTopic + LLM
- **Multi-stage QA Pipeline**: Generate, Select, Verify, Correct for quality assurance
- **Multimodal Support**: Handles text, tables, figures, and images in documents
- **Cross-Document Retrieval**: Unified FAISS index enables retrieval across all documents
- **Hierarchical Deduplication**: Two-stage clustering with LLM-based merging
- **Multiple Backend Support**: Gemini, OpenAI, and local Ollama models
- **Optimized Evaluation**: 3-5x faster metrics with harmonized RAGAS implementation
- **Fully Parallelized**: Thread and process pools for maximum throughput

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Pipeline Overview](#pipeline-overview)
- [Usage](#usage)
- [Output Format](#output-format)
- [Evaluation Metrics](#evaluation-metrics)
- [Hyperparameter Guide](#hyperparameter-guide)
- [Core Modules](#core-modules)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Installation

### From PyPI (Recommended)

```bash
pip install mirage-bench
```

### From Source

```bash
# Clone the repository
git clone https://github.com/ChandanKSahu/MiRAGE.git
cd MiRAGE

# Install in development mode
pip install -e .

# Or install with setup.py
python setup.py install
```

### With Optional Dependencies

```bash
# GPU support (CUDA-enabled embeddings and FAISS)
pip install mirage-bench[gpu]

# PDF processing (Docling for PDF to Markdown conversion)
pip install mirage-bench[pdf]

# Evaluation metrics (RAGAS and LangChain)
pip install mirage-bench[eval]

# Development tools (testing, linting)
pip install mirage-bench[dev]

# All dependencies
pip install mirage-bench[all]
```

### Manual Installation (from requirements.txt)

```bash
git clone https://github.com/ChandanKSahu/MiRAGE.git
cd MiRAGE
pip install -r requirements.txt

# Optional: Install additional dependencies as needed
pip install faiss-gpu bitsandbytes accelerate  # GPU support
pip install docling pypdfium2  # PDF processing
pip install ragas datasets langchain langchain-google-genai  # Evaluation
```

## Quick Start

### 1. Add Your Data

Place your documents in the `data/documents/` folder:

```bash
# Create data directory if it doesn't exist
mkdir -p data/documents

# Add your PDF, HTML, or other supported documents
cp /path/to/your/documents/*.pdf data/documents/
```

**Supported formats:** PDF, HTML, XHTML

### 2. Set Up Configuration

```bash
cp config.yaml.example config.yaml
```

Edit `config.yaml` with your API keys and paths:

```yaml
backend:
  active: GEMINI
  gemini:
    api_key_path: ~/.config/gemini/api_key.txt  # Or set GEMINI_API_KEY env var

paths:
  input_pdf_dir: data/documents  # Your data folder
  output_dir: output/my_dataset
```

### 3. Run Preflight Checks

```bash
python preflight_check.py
```

### 4. Generate QA Dataset

```bash
python main.py
```

### Expected Directory Structure

```
MiRAGE/
├── data/
│   └── documents/          # Add your PDFs/HTMLs here
│       ├── document1.pdf
│       ├── document2.pdf
│       └── ...
├── output/                 # Generated results appear here
│   └── my_dataset/
│       ├── chunks.json
│       ├── qa_deduplicated.json
│       └── ...
├── config.yaml             # Your configuration
└── ...
```

## Configuration

MiRAGE uses a YAML configuration file. Key sections:

| Section | Description |
|---------|-------------|
| `backend` | LLM/VLM provider settings (Gemini, OpenAI, Ollama) |
| `paths` | Input documents and output directory |
| `qa_generation` | Target QA pairs and type (multihop/multimodal/text) |
| `embedding` | Embedding model and batch size |
| `retrieval` | Multi-hop retrieval parameters |
| `deduplication` | Similarity thresholds for deduplication |
| `evaluation` | Metrics and evaluation settings |

See [`config.yaml.example`](config.yaml.example) for full documentation.

### Environment Variables

```bash
export GEMINI_API_KEY="your-key"  # Google Gemini
export OPENAI_API_KEY="your-key"  # OpenAI
```

## Pipeline Overview

The MiRAGE framework operates through a multi-stage pipeline that transforms raw documents into high-quality QA datasets:

```
+------------------------------------------------------------------+
|  STEP 1: Document Processing                                      |
|  PDF/HTML -> Markdown -> Semantic Chunks                          |
+--------------------------------+---------------------------------+
                                 |
                                 v
+------------------------------------------------------------------+
|  STEP 2: Embedding and Indexing                                   |
|  Embed all chunks -> Build unified FAISS index                    |
+--------------------------------+---------------------------------+
                                 |
                                 v
+------------------------------------------------------------------+
|  STEP 3: Domain and Expert Extraction                             |
|  BERTopic analysis -> LLM domain/role identification              |
+--------------------------------+---------------------------------+
                                 |
                                 v
+------------------------------------------------------------------+
|  STEP 4: QA Generation (per chunk, parallel)                      |
|  +--------------------------------------------------------------+ |
|  | 4.1 Verify chunk completeness                                | |
|  | 4.2 Multi-hop retrieval for incomplete chunks                | |
|  | 4.3 Generate QA pairs from complete context                  | |
|  | 4.4 Select high-quality pairs                                | |
|  | 4.5 Verify correctness and context necessity                 | |
|  | 4.6 Correct failed pairs (optional)                          | |
|  +--------------------------------------------------------------+ |
+--------------------------------+---------------------------------+
                                 |
                                 v
+------------------------------------------------------------------+
|  STEP 5: Hierarchical Deduplication                               |
|  Question clustering -> Answer sub-clustering -> LLM merging      |
+--------------------------------+---------------------------------+
                                 |
                                 v
+------------------------------------------------------------------+
|  STEP 6: Evaluation                                               |
|  RAGAS metrics + Custom metrics (faithfulness, relevancy, etc)    |
+------------------------------------------------------------------+
```

## Usage

### Full Pipeline

```bash
python main.py
```

### Individual Components

```bash
# Preflight checks
python preflight_check.py

# QA generation only
python qa_gen_multi_hop.py

# Deduplication only
python deduplication.py

# Evaluation only
python metrics_optimized.py path/to/qa.json output_dir/

# Domain extraction
python domain_expert.py
```

### Programmatic Usage

```python
from main import (
    load_chunks,
    embed_all_chunks,
    get_domain_and_expert,
    generate_qa_dataset_parallel,
    deduplicate_qa_dataset_parallel
)

# Load your chunks
chunks = load_chunks("path/to/chunks.json")

# Embed and index
embeddings_dir, embeddings, chunk_ids = embed_all_chunks(
    chunks, chunks_file, output_dir
)

# Extract domain
domain, expert = get_domain_and_expert(chunks_file, embeddings)

# Generate QA pairs
successful, failed, contexts, stats = generate_qa_dataset_parallel(
    chunks, domain, expert
)

# Deduplicate
deduplicate_qa_dataset_parallel(
    input_file, output_file, expert, domain
)
```

## Output Format

### Sample Generated Question-Answer Pair

MiRAGE generates comprehensive QA pairs with full context traceability:

<p align="center">
  <img src="assets/ample question-answer pair generated.png" alt="Sample Generated QA Pair" width="100%">
</p>

### QA Dataset Structure (qa_deduplicated.json)

```json
[
  {
    "chunk_id": 1,
    "question": "What efficiency must a 75kW IE4 motor achieve?",
    "answer": "A 75kW IE4 motor must achieve 96.0% efficiency at 50Hz...",
    "context_chunks": [...],
    "hop_count": 2,
    "relevance_score": "9",
    "difficulty_score": "7",
    "expert_persona": "Motor Design Engineer",
    "domain": "Electrical Engineering"
  }
]
```

### Evaluation Report (subset_evaluation_report.json)

```json
{
  "ragas_metrics": {
    "faithfulness": 0.85,
    "answer_relevancy": 0.82,
    "context_precision": 0.78,
    "context_recall": 0.80
  },
  "multihop_metrics": {
    "avg_reasoning_score": 0.75
  },
  "multimodal_metrics": {
    "avg_visual_dependency": 0.65
  },
  "context_necessity": {
    "avg_context_necessity_score": 0.88
  }
}
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Faithfulness** | Answer grounded in context |
| **Answer Relevancy** | Answer addresses the question |
| **Context Precision** | Retrieved chunks are relevant |
| **Context Recall** | Context contains reference info |
| **Multi-hop Reasoning** | Quality of multi-step reasoning |
| **Visual Dependency** | Requires image to answer |
| **Context Necessity** | Requires context (anti-parametric bias) |
| **Domain Coverage** | Corpus coverage |

## Hyperparameter Guide

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_depth` | 10 | Maximum retrieval iterations |
| `max_breadth` | 5 | Search queries per iteration |
| `chunks_per_search` | 2 | Chunks retrieved per query |
| `qa_max_workers` | 6 | Parallel workers for QA gen |
| `question_similarity_threshold` | 0.75 | Question clustering threshold |

### Recommended Settings

| Use Case | max_depth | max_breadth | chunks_per_search |
|----------|-----------|-------------|-------------------|
| **Quick Testing** | 2 | 2 | 1 |
| **Balanced (Default)** | 10 | 5 | 2 |
| **Thorough** | 20 | 10 | 3 |

## Core Modules

| Module | Description |
|--------|-------------|
| `main.py` | Pipeline orchestration |
| `config_loader.py` | Configuration management |
| `preflight_check.py` | Service validation |
| `context_retrieved.py` | Multi-hop retrieval |
| `qa_gen_multi_hop.py` | QA generation and verification |
| `deduplication.py` | Hierarchical deduplication |
| `domain_expert.py` | Domain/expert extraction |
| `embed_models.py` | Embedding model interfaces |
| `call_llm.py` | LLM/VLM API interface |
| `metrics_optimized.py` | Evaluation metrics |
| `prompt.py` | Prompt templates |

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use MiRAGE in your research, please cite:

```bibtex
@software{mirage2024,
  title = {MiRAGE: A Multiagent Framework for Generating Multimodal Multihop QA Datasets for RAG Evaluation},
  author = {MiRAGE Authors},
  year = {2024},
  url = {https://github.com/ChandanKSahu/MiRAGE}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [RAGAS](https://github.com/explodinggradients/ragas) for evaluation metrics
- [BERTopic](https://github.com/MaartenGr/BERTopic) for topic modeling
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for similarity search
- [Docling](https://github.com/DS4SD/docling) for PDF processing
