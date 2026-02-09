"""
MiRAGE Results Analysis Example
================================

Shows how to load, filter, analyze, and export QA datasets.
You don't need to run the pipeline — just load existing results!

Prerequisites:
    pip install mirage-benchmark

Usage:
    python 04_results_analysis.py
"""

from mirage import MiRAGEResults

# ---------------------------------------------------------------------------
# Load existing results
# ---------------------------------------------------------------------------
# Load QA pairs from a previously generated dataset.
# Supports both JSON and JSONL formats.

results = MiRAGEResults.load("output/my_dataset/qa_multihop_pass.json")
print(f"Loaded {len(results)} QA pairs")

# ---------------------------------------------------------------------------
# Browse the data
# ---------------------------------------------------------------------------
# Results are iterable — loop through them like a list:

for qa in results[:3]:  # First 3 QA pairs
    print(f"Q: {qa['question']}")
    print(f"A: {qa['answer']}")
    print()

# Quick access to all questions and answers:
print(f"All questions: {results.questions[:3]}...")
print(f"All answers: {results.answers[:3]}...")

# ---------------------------------------------------------------------------
# Filter QA pairs
# ---------------------------------------------------------------------------
# Filter by any field in the QA pair dictionary:

multihop_only = results.filter(question_type="multihop")
print(f"\nMulti-hop questions: {len(multihop_only)}")

hard_questions = results.filter(difficulty="hard")
print(f"Hard questions: {len(hard_questions)}")

# ---------------------------------------------------------------------------
# Random sampling
# ---------------------------------------------------------------------------
# Take a random subset (useful for evaluation or manual review):

sample = results.sample(n=5, seed=42)
print(f"\nRandom sample of 5:")
for qa in sample:
    print(f"  - {qa['question'][:80]}...")

# ---------------------------------------------------------------------------
# Export to different formats
# ---------------------------------------------------------------------------

# JSON (pretty-printed)
results.save("exported_dataset.json")

# JSONL (one JSON object per line — great for streaming)
results.save("exported_dataset.jsonl", format="jsonl")

# pandas DataFrame (for analysis in Jupyter notebooks, etc.)
df = results.to_dataframe()
print(f"\nDataFrame columns: {list(df.columns)}")
print(f"DataFrame shape: {df.shape}")

# Save as CSV
df.to_csv("exported_dataset.csv", index=False)
print("Exported to CSV, JSON, and JSONL")

# Dictionary format
d = results.to_dict()
print(f"\nDict keys: {list(d.keys())}")
