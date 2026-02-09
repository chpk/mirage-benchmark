"""
MiRAGE Config File Example
===========================

Load pipeline settings from a YAML config file.
Useful for reproducible experiments and team workflows.

Prerequisites:
    pip install mirage-benchmark

Usage:
    # First, generate a config file:
    run_mirage --init-config

    # Then edit config.yaml with your settings, and run:
    python 03_from_config_file.py
"""

from mirage import MiRAGE

# ---------------------------------------------------------------------------
# Option A: Load from an existing config.yaml
# ---------------------------------------------------------------------------
# If you have a config.yaml (generated via `run_mirage --init-config`),
# load it directly:

pipeline = MiRAGE.from_config(
    "config.yaml",                   # Path to your config file
    api_key="YOUR_API_KEY",          # Override API key
    num_qa_pairs=50,                 # Override number of QA pairs
)

# You can also override any parameter after loading:
pipeline.configure(
    output_dir="output/from_config",
    max_depth=2,
)

results = pipeline.run()
print(f"Generated {len(results)} QA pairs from config file")

# ---------------------------------------------------------------------------
# Option B: Save your configuration for later
# ---------------------------------------------------------------------------
# Save the current pipeline config so others can reproduce your setup:

pipeline.save_config("my_experiment_config.yaml")
print("Config saved to my_experiment_config.yaml")

# Later, anyone can reproduce with:
#   pipeline = MiRAGE.from_config("my_experiment_config.yaml", api_key="their-key")
#   results = pipeline.run()
