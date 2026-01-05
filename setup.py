#!/usr/bin/env python3
"""
MiRAGE - Multimodal Multihop RAG Evaluation Dataset Generator

A multi-agent framework for generating high-quality, multimodal, multihop 
question-answer datasets for evaluating Retrieval-Augmented Generation (RAG) systems.
"""

import os
from setuptools import setup, find_packages

# Read the README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    requirements = []
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#') and not line.startswith('-'):
                    requirements.append(line)
    return requirements

# Core dependencies
INSTALL_REQUIRES = [
    "torch>=2.0.0",
    "numpy>=1.21.0",
    "Pillow>=9.0.0",
    "transformers>=4.44.0",
    "huggingface_hub>=0.16.0",
    "sentence-transformers>=2.2.0",
    "faiss-cpu>=1.7.0",
    "tqdm>=4.65.0",
    "PyYAML>=6.0",
    "requests>=2.28.0",
    "aiohttp>=3.8.0",
    "bertopic>=0.16.0",
    "umap-learn>=0.5.0",
    "scikit-learn>=1.0.0",
]

# Optional dependencies
EXTRAS_REQUIRE = {
    "gpu": [
        "faiss-gpu>=1.7.0",
        "bitsandbytes>=0.43.0",
        "accelerate>=0.20.0",
    ],
    "pdf": [
        "docling>=0.1.0",
        "pypdfium2>=4.0.0",
    ],
    "eval": [
        "ragas>=0.1.0",
        "datasets>=2.0.0",
        "langchain>=0.1.0",
        "langchain-google-genai>=1.0.0",
        "langchain-openai>=0.1.0",
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
    ],
}

# Combined 'all' extra
EXTRAS_REQUIRE["all"] = (
    EXTRAS_REQUIRE["gpu"] + 
    EXTRAS_REQUIRE["pdf"] + 
    EXTRAS_REQUIRE["eval"] + 
    EXTRAS_REQUIRE["dev"]
)

setup(
    name="mirage-bench",
    version="1.0.0",
    author="MiRAGE Authors",
    author_email="",
    description="MiRAGE: A Multiagent Framework for Generating Multimodal Multihop QA Datasets for RAG Evaluation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/mirage-bench",
    project_urls={
        "Documentation": "https://github.com/your-org/mirage-bench#readme",
        "Source": "https://github.com/your-org/mirage-bench",
        "Issues": "https://github.com/your-org/mirage-bench/issues",
    },
    license="Apache-2.0",
    
    # Package discovery
    packages=find_packages(exclude=["tests", "tests.*", "archive", "archive.*"]),
    py_modules=[
        "main",
        "call_llm",
        "config_loader",
        "context_retrieved",
        "data_stats",
        "deduplication",
        "domain_expert",
        "embed_models",
        "md_to_semantic_chunks",
        "metrics",
        "metrics_optimized",
        "pdf_to_md",
        "preflight_check",
        "prompt",
        "qa_gen_multi_hop",
        "rerankers_multimodal",
        "rerankers_text_qa_llm",
        "run_ablation_study",
    ],
    
    # Include non-Python files
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.md", "*.txt"],
    },
    
    # Dependencies
    python_requires=">=3.9",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    
    # Entry points for CLI commands
    entry_points={
        "console_scripts": [
            "mirage=main:main",
            "mirage-preflight=preflight_check:main",
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    
    # Keywords for PyPI search
    keywords=[
        "rag",
        "retrieval-augmented-generation",
        "question-answering",
        "multimodal",
        "evaluation",
        "benchmark",
        "nlp",
        "llm",
        "vlm",
        "dataset-generation",
    ],
)
