"""
Pipeline module for MiRAGE - Document processing, QA generation, and deduplication.
"""

from mirage.pipeline.pdf_processor import (
    process_pdf_to_markdown,
    process_directory,
)

from mirage.pipeline.chunker import (
    chunk_markdown_to_semantic,
    process_markdown_file,
)

from mirage.pipeline.context import (
    build_complete_context,
    retrieve_similar_chunks,
    ContextBuilder,
)

from mirage.pipeline.qa_generator import (
    generate_qa_for_chunk,
    verify_qa_pair,
    select_best_qa_pairs,
)

from mirage.pipeline.domain import (
    fetch_domain_and_role,
    load_domain_expert_from_env,
    save_domain_expert_to_env,
    DomainExtractor,
)

from mirage.pipeline.deduplication import (
    deduplicate_qa_pairs,
    cluster_questions,
    merge_similar_qa,
)

__all__ = [
    # PDF Processing
    "process_pdf_to_markdown",
    "process_directory",
    # Chunking
    "chunk_markdown_to_semantic",
    "process_markdown_file",
    # Context
    "build_complete_context",
    "retrieve_similar_chunks",
    "ContextBuilder",
    # QA Generation
    "generate_qa_for_chunk",
    "verify_qa_pair",
    "select_best_qa_pairs",
    # Domain
    "fetch_domain_and_role",
    "load_domain_expert_from_env",
    "save_domain_expert_to_env",
    "DomainExtractor",
    # Deduplication
    "deduplicate_qa_pairs",
    "cluster_questions",
    "merge_similar_qa",
]
