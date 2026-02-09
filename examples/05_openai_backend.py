"""
MiRAGE with OpenAI Backend
============================

Use OpenAI models (GPT-4o, GPT-4o-mini) instead of Gemini.

Prerequisites:
    pip install mirage-benchmark
    export OPENAI_API_KEY="your-openai-api-key"

Usage:
    python 05_openai_backend.py
"""

from mirage import MiRAGE

# ---------------------------------------------------------------------------
# Using OpenAI as the LLM backend
# ---------------------------------------------------------------------------
# Just change backend="openai" and provide your OpenAI API key.
# The pipeline works exactly the same way — only the LLM calls change.

pipeline = MiRAGE(
    input_dir="data/my_documents",
    output_dir="output/openai_results",
    backend="openai",                    # Use OpenAI instead of Gemini
    api_key="YOUR_OPENAI_API_KEY",       # Or set OPENAI_API_KEY env variable
    # llm_model="gpt-4o-mini",           # Default LLM for OpenAI
    # vlm_model="gpt-4o",               # Default VLM for OpenAI
    num_qa_pairs=10,
)

results = pipeline.run()
print(f"Generated {len(results)} QA pairs using OpenAI")
results.save("output/openai_results/dataset.json")
