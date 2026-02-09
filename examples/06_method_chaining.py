"""
MiRAGE Method Chaining Example
================================

Shows the fluent API style — configure and run in one expression.

Prerequisites:
    pip install mirage-benchmark

Usage:
    python 06_method_chaining.py
"""

import os
from mirage import MiRAGE, MiRAGEConfig

# ---------------------------------------------------------------------------
# One-liner: configure and run
# ---------------------------------------------------------------------------
# Method chaining lets you build and run the pipeline in a single expression:

results = (
    MiRAGE(
        input_dir="data/my_documents",
        output_dir="output/chained",
        backend="gemini",
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    .configure(num_qa_pairs=20, max_depth=2)
    .run()
)

print(f"Generated {len(results)} QA pairs")
results.save("output/chained/dataset.json")

# ---------------------------------------------------------------------------
# Build config programmatically
# ---------------------------------------------------------------------------
# Create a config from a dictionary (e.g., from a JSON API request):

user_settings = {
    "backend": "gemini",
    "api_key": os.environ.get("GEMINI_API_KEY"),
    "input_dir": "data/my_documents",
    "output_dir": "output/from_dict",
    "num_qa_pairs": 30,
    "max_depth": 2,
    "embedding_model": "gemini",
}

config = MiRAGEConfig.from_dict(user_settings)
pipeline = MiRAGE(config=config)
print(f"Pipeline: {pipeline}")

# ---------------------------------------------------------------------------
# Preflight check before running
# ---------------------------------------------------------------------------
# Validate your configuration before committing to a long run:

if pipeline.preflight():
    print("All checks passed! Running pipeline...")
    results = pipeline.run()
else:
    print("Configuration issues detected. Fix them before running.")
