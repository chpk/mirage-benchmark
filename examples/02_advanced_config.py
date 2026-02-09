"""
MiRAGE Advanced Configuration Example
======================================

Shows how to configure every aspect of the pipeline:
models, devices, processing options, and more.

Prerequisites:
    pip install mirage-benchmark
    export GEMINI_API_KEY="your-gemini-api-key"

Usage:
    python 02_advanced_config.py
"""

from mirage import MiRAGE

# ---------------------------------------------------------------------------
# Full pipeline configuration
# ---------------------------------------------------------------------------
# MiRAGE exposes 47 configuration parameters. Here are the most useful ones.

pipeline = MiRAGE(
    # --- Where to read and write ---
    input_dir="data/research_papers",
    output_dir="output/advanced_run",

    # --- LLM Backend ---
    backend="gemini",                    # "gemini", "openai", or "ollama"
    api_key="YOUR_API_KEY",              # Or use environment variables
    # llm_model="gemini-2.0-flash",     # Override default LLM model
    # vlm_model="gemini-2.0-flash",     # Override default VLM model

    # --- Embedding Model ---
    # Controls how document chunks are embedded for similarity search.
    # Options: "auto", "nomic", "bge_m3", "gemini", "qwen3_vl", "qwen2_vl"
    embedding_model="auto",

    # --- Reranker Model ---
    # Re-ranks retrieved chunks for better context selection.
    # Options: "gemini_vlm", "monovlm", "text_embedding"
    reranker_model="gemini_vlm",

    # --- QA Generation ---
    num_qa_pairs=100,                    # Target number of QA pairs
    qa_type="multihop",                  # "multihop", "multimodal", "text", "mix"
    max_depth=3,                         # How many hops for multi-hop questions (1-20)
    max_breadth=5,                       # Search breadth per hop (1-10)

    # --- Document Processing ---
    ocr_engine="easyocr",                # OCR for scanned PDFs
    do_table_structure=True,             # Extract table structures
    do_ocr=True,                         # Enable OCR
    image_resolution_scale=2.0,          # Image quality (1.0-4.0)

    # --- Device & GPU ---
    device="cuda:0",                     # "cuda", "cuda:0", "cpu", or None (auto)
    embedding_gpus=[0],                  # Which GPUs for embeddings

    # --- Performance ---
    max_workers=4,                       # Parallel QA generation workers
    embed_batch_size=16,                 # Embedding batch size
    requests_per_minute=60,              # API rate limit

    # --- Optional Pipeline Steps ---
    run_deduplication=True,              # Remove duplicate QA pairs
    run_evaluation=False,                # Compute quality metrics (needs [eval] extras)
)

# ---------------------------------------------------------------------------
# Run and inspect results
# ---------------------------------------------------------------------------
results = pipeline.run()

print(f"Generated {len(results)} QA pairs")
print(f"From {len(results.chunks)} document chunks")
print(f"Stats: {results.stats}")

# Save in different formats
results.save("output/advanced_run/dataset.json")              # JSON
results.save("output/advanced_run/dataset.jsonl", format="jsonl")  # JSONL

# Convert to pandas DataFrame for analysis
df = results.to_dataframe()
print(f"\nDataFrame shape: {df.shape}")
print(df.head())
