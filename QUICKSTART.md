# MiRAGE Quick Start Guide

## Installation

```bash
pip install mirage-benchmark
```

## Basic Usage

### 1. Set up API Key

**For Gemini (default):**
```bash
export GEMINI_API_KEY="your-gemini-api-key-here"
```

**For OpenAI:**
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
export MIRAGE_BACKEND="OPENAI"  # Optional: defaults to GEMINI
```

### 2. Run the Pipeline

**Basic command:**
```bash
run_mirage --input data/sample --output results/my_dataset --num-qa-pairs 1
```

**With all options:**
```bash
run_mirage \
  --input data/sample \
  --output results/my_dataset \
  --num-qa-pairs 1 \
  --backend gemini \
  --api-key YOUR_API_KEY \
  --max-depth 2 \
  --embedding-model auto \
  --reranker-model gemini_vlm
```

## Command-Line Options

### Required Arguments
- `--input, -i`: Input directory containing PDF/HTML documents
- `--output, -o`: Output directory for generated QA dataset

### Optional Arguments
- `--num-qa-pairs`: Number of QA pairs to generate (default: 100)
- `--backend, -b`: LLM backend: `gemini` (default), `openai`, or `ollama`
- `--api-key, -k`: API key for the selected backend
- `--max-depth`: Maximum depth for multi-hop retrieval (default: 2)
- `--embedding-model`: Embedding model: `auto` (default), `qwen3_vl`, `nomic`, or `bge_m3`
- `--reranker-model`: Reranker model: `gemini_vlm` (default), `monovlm`, or `text_embedding`
- `--deduplication`: Enable QA deduplication (off by default)
- `--evaluation`: Enable evaluation metrics (off by default)
- `--verbose, -v`: Enable verbose output

## Examples

### Example 1: Generate 1 QA pair with default settings
```bash
export GEMINI_API_KEY="your-key"
run_mirage --input data/sample --output results/test --num-qa-pairs 1
```

### Example 2: Use OpenAI backend
```bash
export OPENAI_API_KEY="your-key"
run_mirage --input data/documents --output results/openai_test \
  --backend openai --num-qa-pairs 10
```

### Example 3: Generate with deduplication and evaluation
```bash
run_mirage --input data/documents --output results/full \
  --num-qa-pairs 50 --deduplication --evaluation
```

### Example 4: Custom embedding and reranker models
```bash
run_mirage --input data/documents --output results/custom \
  --num-qa-pairs 20 \
  --embedding-model nomic \
  --reranker-model monovlm
```

## Output Files

After running, you'll find in the output directory:
- `chunks.json`: Semantic chunks from your documents
- `qa_multihop_pass.json`: Successfully generated QA pairs
- `qa_multihop_fail.json`: Failed QA generation attempts
- `multihop_visualization.html`: Interactive visualization of QA pairs
- `embeddings/`: FAISS index and embeddings
- `markdown/`: Converted markdown files and images

## Troubleshooting

### Command not found
If `run_mirage` command is not found:
```bash
# Check if package is installed
pip show mirage-benchmark

# Reinstall if needed
pip install --upgrade mirage-benchmark
```

### API Key Issues
```bash
# Verify API key is set
echo $GEMINI_API_KEY  # or $OPENAI_API_KEY

# Set it if missing
export GEMINI_API_KEY="your-key"
```

### Preflight Checks
Run preflight checks to verify setup:
```bash
run_mirage --preflight
```

## Auto-Selected Reranker

The reranker is automatically selected based on your backend/API keys:
- **Gemini backend/key** → Uses Gemini VLM reranker (fast, API-based)
- **OpenAI backend** → Uses Gemini VLM if Gemini key available, else MonoVLM
- **No API keys** → Falls back to MonoVLM (local model, slower)

You can override with `--reranker-model` flag.
