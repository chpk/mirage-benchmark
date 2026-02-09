"""
MiRAGE Quick Start Example
==========================

The simplest way to generate a QA dataset from your documents.

Prerequisites:
    pip install mirage-benchmark
    export GEMINI_API_KEY="your-gemini-api-key"

Usage:
    python 01_quick_start.py
"""

from mirage import MiRAGE

# ---------------------------------------------------------------------------
# Step 1: Create a pipeline
# ---------------------------------------------------------------------------
# Point MiRAGE at your documents and choose a backend.
# Gemini is the default and recommended backend.

pipeline = MiRAGE(
    input_dir="data/my_documents",       # Folder with PDFs, HTML, etc.
    output_dir="output/quick_start",     # Where results will be saved
    backend="gemini",                    # LLM backend: "gemini", "openai", "ollama"
    api_key="YOUR_GEMINI_API_KEY",       # Or set GEMINI_API_KEY env variable
    num_qa_pairs=10,                     # How many QA pairs to generate
)

# ---------------------------------------------------------------------------
# Step 2: Run the pipeline
# ---------------------------------------------------------------------------
# This will:
#   1. Convert PDFs/HTML to Markdown
#   2. Semantically chunk the documents
#   3. Build embeddings and FAISS index
#   4. Generate multi-hop QA pairs using the LLM
#
# Returns a MiRAGEResults object with all generated QA pairs.

results = pipeline.run()

# ---------------------------------------------------------------------------
# Step 3: Use the results
# ---------------------------------------------------------------------------

# How many QA pairs were generated?
print(f"\nGenerated {len(results)} QA pairs!\n")

# Print each question-answer pair
for i, qa in enumerate(results, 1):
    print(f"--- QA Pair {i} ---")
    print(f"Q: {qa['question']}")
    print(f"A: {qa['answer']}")
    print()

# Save to a JSON file
results.save("output/quick_start/my_dataset.json")
print("Dataset saved to output/quick_start/my_dataset.json")
