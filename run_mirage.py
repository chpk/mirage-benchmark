#!/usr/bin/env python3
"""
MiRAGE - Main Entry Point

This script runs the complete MiRAGE pipeline for generating multimodal
multihop QA datasets from your documents.

Usage:
    python run_mirage.py                    # Run full pipeline
    python run_mirage.py --preflight        # Run preflight checks only
    python run_mirage.py --config my.yaml   # Use custom config file

Configuration:
    1. Copy config.yaml.example to config.yaml
    2. Add your API keys (Gemini, OpenAI, or configure Ollama)
    3. Set input_pdf_dir to your documents folder
    4. Set output_dir for results

For more information, see README.md
"""

import os
import sys
import argparse
import json
import logging
import numpy as np
import faiss
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

# Add src to path for package imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mirage.core.llm import setup_logging, BACKEND, LLM_MODEL_NAME, VLM_MODEL_NAME, GEMINI_RPM, GEMINI_BURST
from mirage.utils.stats import compute_dataset_stats, print_dataset_stats, compute_qa_category_stats, print_qa_category_stats
from mirage.utils.preflight import run_preflight_checks


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MiRAGE: Multimodal Multihop RAG Evaluation Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run preflight checks only"
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip preflight checks"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()


def main():
    """Main entry point for MiRAGE pipeline."""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("MiRAGE: Multimodal Multihop RAG Evaluation Dataset Generator")
    logger.info("=" * 60)
    logger.info(f"Backend: {BACKEND}")
    logger.info(f"LLM Model: {LLM_MODEL_NAME}")
    logger.info(f"VLM Model: {VLM_MODEL_NAME}")
    
    # Run preflight checks
    if args.preflight:
        logger.info("\nRunning preflight checks...")
        success = run_preflight_checks()
        sys.exit(0 if success else 1)
    
    if not args.skip_preflight:
        logger.info("\nRunning preflight checks...")
        if not run_preflight_checks():
            logger.error("Preflight checks failed. Fix issues above or use --skip-preflight to bypass.")
            sys.exit(1)
        logger.info("Preflight checks passed!\n")
    
    # Import pipeline modules (after preflight to catch import errors)
    from mirage.pipeline.pdf_processor import process_directory as process_pdfs
    from mirage.pipeline.chunker import process_markdown_directory
    from mirage.pipeline.domain import fetch_domain_and_role
    from mirage.pipeline.qa_generator import generate_qa_for_chunk
    from mirage.pipeline.deduplication import deduplicate_qa_dataset
    from mirage.core.config import load_config
    
    # Load configuration
    config = load_config(args.config)
    paths = config.get('paths', {})
    
    input_dir = paths.get('input_pdf_dir', 'data/documents')
    output_dir = paths.get('output_dir', 'output/results')
    
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Process PDFs to Markdown
    logger.info("\n" + "=" * 60)
    logger.info("Step 1: Processing documents to Markdown")
    logger.info("=" * 60)
    markdown_dir = os.path.join(output_dir, "markdown")
    # process_pdfs(input_dir, markdown_dir)
    logger.info(f"Markdown files saved to: {markdown_dir}")
    
    # Step 2: Chunk Markdown to Semantic Chunks
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Creating semantic chunks")
    logger.info("=" * 60)
    chunks_file = os.path.join(output_dir, "chunks.json")
    # process_markdown_directory(markdown_dir, chunks_file)
    logger.info(f"Chunks saved to: {chunks_file}")
    
    # Step 3: Extract Domain and Expert Role
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Extracting domain and expert role")
    logger.info("=" * 60)
    # domain, expert = fetch_domain_and_role(chunks_file)
    # logger.info(f"Domain: {domain}")
    # logger.info(f"Expert: {expert}")
    
    # Step 4: Generate QA Pairs
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Generating QA pairs")
    logger.info("=" * 60)
    qa_file = os.path.join(output_dir, "qa_dataset.json")
    # Generate QA pairs here
    logger.info(f"QA pairs saved to: {qa_file}")
    
    # Step 5: Deduplicate
    logger.info("\n" + "=" * 60)
    logger.info("Step 5: Deduplicating QA pairs")
    logger.info("=" * 60)
    dedup_file = os.path.join(output_dir, "qa_deduplicated.json")
    # deduplicate_qa_dataset(qa_file, dedup_file)
    logger.info(f"Deduplicated QA saved to: {dedup_file}")
    
    # Step 6: Statistics
    logger.info("\n" + "=" * 60)
    logger.info("Step 6: Computing statistics")
    logger.info("=" * 60)
    # stats = compute_dataset_stats(output_dir)
    # print_dataset_stats(stats)
    
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    # Use spawn method for multiprocessing (required for CUDA)
    mp.set_start_method('spawn', force=True)
    main()
