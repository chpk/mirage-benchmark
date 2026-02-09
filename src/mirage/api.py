"""
MiRAGE High-Level Python API

A clean, library-style interface for MiRAGE — designed to feel like
HuggingFace Transformers, OpenAI, or scikit-learn.

===========================================================================
Quick Start:
===========================================================================

    from mirage import MiRAGE

    pipeline = MiRAGE(
        input_dir="data/my_documents",
        output_dir="output/results",
        backend="gemini",
        api_key="your-api-key",
    )
    results = pipeline.run()
    results.save("my_dataset.json")

===========================================================================
Step-by-Step Pipeline:
===========================================================================

    pipeline = MiRAGE(backend="gemini", api_key="key", input_dir="docs", output_dir="out")

    # Run individual steps
    chunks = pipeline.convert_and_chunk()       # PDF -> Markdown -> Chunks
    pipeline.embed(chunks)                      # Build FAISS index
    results = pipeline.generate(chunks)         # Generate QA pairs
    results = pipeline.deduplicate(results)     # Remove duplicates
    results.save("dataset.json")

===========================================================================
Configuration:
===========================================================================

    from mirage import MiRAGE, MiRAGEConfig

    # Full control
    config = MiRAGEConfig(
        backend="gemini",
        api_key="key",
        num_qa_pairs=200,
        embedding_model="nomic",
        ocr_engine="easyocr",
        device="cuda:0",
    )
    pipeline = MiRAGE(config=config)
    results = pipeline.run()

    # From YAML
    pipeline = MiRAGE.from_config("config.yaml", num_qa_pairs=50)

===========================================================================
Results:
===========================================================================

    results = pipeline.run()

    for qa in results:                    # Iterate
        print(qa['question'])

    results.save("dataset.json")          # Save as JSON
    results.save("dataset.jsonl", format="jsonl")  # Save as JSONL
    df = results.to_dataframe()           # pandas DataFrame
    d  = results.to_dict()                # dict

    loaded = MiRAGEResults.load("dataset.json")  # Load from file
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class MiRAGEConfig:
    """
    Complete configuration for the MiRAGE pipeline.
    
    All parameters have sensible defaults. At minimum, you need:
    - ``input_dir``: Path to documents (PDFs, HTML, etc.)
    - ``output_dir``: Where to save results
    - ``backend``: LLM backend ("gemini", "openai", "ollama")
    - ``api_key``: API key for the chosen backend
    
    Example::
    
        config = MiRAGEConfig(
            input_dir="data/docs",
            output_dir="output",
            backend="gemini",
            api_key="your-key",
            num_qa_pairs=50,
            embedding_model="nomic",
            device="cuda:0",
        )
    """
    # =========================================================================
    # Core / Required
    # =========================================================================
    input_dir: str = "data/documents"
    output_dir: str = "output/results"
    backend: str = "gemini"
    api_key: Optional[str] = None
    
    # =========================================================================
    # Model Selection
    # =========================================================================
    llm_model: Optional[str] = None
    """LLM model name. Auto-selected per backend if None.
    Gemini: gemini-2.0-flash, OpenAI: gpt-4o-mini, Ollama: llama3"""
    
    vlm_model: Optional[str] = None
    """Vision-Language model name. Auto-selected per backend if None.
    Gemini: gemini-2.0-flash, OpenAI: gpt-4o, Ollama: llava"""
    
    embedding_model: str = "auto"
    """Embedding model: "auto", "nomic", "bge_m3", "gemini", "qwen3_vl", "qwen2_vl" """
    
    reranker_model: str = "gemini_vlm"
    """Reranker model: "gemini_vlm", "monovlm", "text_embedding" """
    
    # =========================================================================
    # QA Generation
    # =========================================================================
    num_qa_pairs: int = 10
    """Target number of QA pairs to generate."""
    
    qa_type: str = "multihop"
    """QA type: "multihop", "multimodal", "text", "mix" """
    
    max_depth: int = 2
    """Maximum depth for multi-hop context retrieval (1-20). Higher = more hops but more API calls."""
    
    max_breadth: int = 5
    """Maximum breadth for search verification (1-10). Higher = wider search per hop."""
    
    chunks_per_search: int = 2
    """Number of chunks retrieved per search query."""
    
    chunk_addition_mode: str = "RELATED"
    """How to add chunks during context building: "RELATED" or "EXPLANATORY" """
    
    # =========================================================================
    # Document Processing (PDF/HTML -> Markdown)
    # =========================================================================
    ocr_engine: str = "easyocr"
    """OCR engine for scanned documents: "easyocr" (default).
    EasyOCR supports 80+ languages and works on CPU and GPU."""
    
    image_resolution_scale: float = 2.0
    """Scale factor for image resolution during PDF processing (1.0-4.0)."""
    
    do_ocr: bool = True
    """Enable OCR for scanned PDFs."""
    
    do_table_structure: bool = True
    """Enable table structure extraction."""
    
    do_code_enrichment: bool = True
    """Enable code block enrichment."""
    
    do_formula_enrichment: bool = True
    """Enable formula/equation enrichment."""
    
    # =========================================================================
    # Chunking
    # =========================================================================
    chunk_window_size: int = 20000
    """Window size in characters for semantic chunking (default: 20000 ~ 5000 tokens)."""
    
    chunk_overlap_size: int = 2000
    """Overlap between chunking windows in characters (default: 2000 ~ 500 tokens)."""
    
    # =========================================================================
    # Embeddings & Indexing
    # =========================================================================
    embed_batch_size: int = 16
    """Batch size for embedding generation."""
    
    cache_embeddings: bool = True
    """Cache embeddings to disk for reuse across runs."""
    
    faiss_use_gpu: bool = False
    """Use GPU for FAISS similarity search (requires faiss-gpu via conda)."""
    
    faiss_gpu_id: int = 0
    """GPU ID for FAISS (if faiss_use_gpu=True)."""
    
    # =========================================================================
    # Parallel Processing
    # =========================================================================
    max_workers: int = 4
    """Number of parallel workers for QA generation."""
    
    num_cpu_workers: int = 3
    """Number of CPU workers for document processing."""
    
    dedup_max_workers: int = 4
    """Number of workers for deduplication."""
    
    # =========================================================================
    # Rate Limiting
    # =========================================================================
    requests_per_minute: int = 60
    """API rate limit (requests per minute)."""
    
    burst_size: int = 15
    """Burst size for rate limiting."""
    
    # =========================================================================
    # Device & GPU
    # =========================================================================
    device: Optional[str] = None
    """Device for model inference: "cuda", "cuda:0", "cuda:1", "cpu", or None (auto-detect).
    When None, GPU is used if available, otherwise CPU."""
    
    embedding_gpus: Optional[List[int]] = None
    """List of GPU IDs for embedding model (e.g., [0], [0,1]). None = auto."""
    
    pdf_processing_gpus: Optional[List[int]] = None
    """List of GPU IDs for PDF processing workers. None = auto."""
    
    # =========================================================================
    # Pipeline Control
    # =========================================================================
    skip_pdf_processing: bool = False
    """Skip PDF-to-Markdown conversion (use existing markdown files)."""
    
    skip_chunking: bool = False
    """Skip semantic chunking (use existing chunks.json)."""
    
    run_deduplication: bool = False
    """Enable QA pair deduplication after generation."""
    
    run_evaluation: bool = False
    """Enable evaluation metrics computation (requires [eval] extras)."""
    
    enable_checkpointing: bool = True
    """Enable checkpoint/resume for long-running pipelines."""
    
    max_pages: Optional[int] = None
    """Maximum number of pages to process. None = no limit."""
    
    max_pdfs: Optional[int] = None
    """Maximum number of PDF files to process. None = no limit."""
    
    sort_by_size: bool = True
    """Sort documents by file size (smallest first) for processing."""
    
    # =========================================================================
    # QA Correction
    # =========================================================================
    qa_correction_enabled: bool = True
    """Enable automatic QA correction pass."""
    
    qa_correction_max_attempts: int = 1
    """Maximum correction attempts per QA pair."""
    
    # =========================================================================
    # Deduplication
    # =========================================================================
    dedup_alpha: float = 0.6
    """Alpha parameter for deduplication scoring."""
    
    dedup_question_similarity_threshold: float = 0.75
    """Similarity threshold for question deduplication."""
    
    # =========================================================================
    # Advanced
    # =========================================================================
    config_file: Optional[str] = None
    """Path to config.yaml for full YAML-based configuration. 
    When provided, YAML values are used as base, and other parameters override them."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MiRAGEConfig":
        """Create configuration from a dictionary.
        
        Unknown keys are silently ignored, making it safe to pass
        partial or extra configurations.
        """
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)
    
    @classmethod
    def from_yaml(cls, path: str) -> "MiRAGEConfig":
        """
        Load configuration from a YAML file (config.yaml format).
        
        Maps the nested YAML structure to flat MiRAGEConfig fields.
        
        Args:
            path: Path to config.yaml
        
        Returns:
            MiRAGEConfig instance.
        """
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        
        # Map YAML structure to flat config
        backend_cfg = data.get('backend', {})
        active = backend_cfg.get('active', 'GEMINI').lower()
        active_cfg = backend_cfg.get(active, {})
        paths = data.get('paths', {})
        processing = data.get('processing', {})
        parallel = data.get('parallel', {})
        embedding = data.get('embedding', {})
        retrieval = data.get('retrieval', {}).get('multihop', {})
        qa_gen = data.get('qa_generation', {})
        rate_limit = data.get('rate_limiting', {})
        pdf = data.get('pdf_processing', {})
        dedup = data.get('deduplication', {})
        qa_corr = data.get('qa_correction', {})
        faiss_cfg = data.get('faiss', {})
        eval_cfg = data.get('evaluation', {})
        
        return cls(
            # Core
            input_dir=paths.get('input_pdf_dir', 'data/documents'),
            output_dir=paths.get('output_dir', 'output/results'),
            backend=active,
            llm_model=active_cfg.get('llm_model'),
            vlm_model=active_cfg.get('vlm_model'),
            # Models
            embedding_model=embedding.get('model', 'auto'),
            # QA Generation
            num_qa_pairs=qa_gen.get('num_qa_pairs', 10),
            qa_type=qa_gen.get('type', 'multihop'),
            max_depth=retrieval.get('max_depth', 2),
            max_breadth=retrieval.get('max_breadth', 5),
            chunks_per_search=retrieval.get('chunks_per_search', 2),
            chunk_addition_mode=retrieval.get('chunk_addition_mode', 'RELATED'),
            # Processing
            max_workers=parallel.get('qa_max_workers', 4),
            num_cpu_workers=parallel.get('num_workers', 3),
            dedup_max_workers=parallel.get('dedup_max_workers', 4),
            embed_batch_size=embedding.get('batch_size', 16),
            embedding_gpus=embedding.get('gpus'),
            max_pdfs=processing.get('max_pdfs'),
            sort_by_size=processing.get('sort_by_size', True),
            # PDF processing
            image_resolution_scale=pdf.get('image_resolution_scale', 2.0),
            # Rate limiting
            requests_per_minute=rate_limit.get('requests_per_minute', 60),
            burst_size=rate_limit.get('burst_size', 15),
            # FAISS
            faiss_use_gpu=faiss_cfg.get('use_gpu', False),
            faiss_gpu_id=faiss_cfg.get('gpu_id', 0),
            # Dedup
            dedup_alpha=dedup.get('alpha', 0.6),
            dedup_question_similarity_threshold=dedup.get('question_similarity_threshold', 0.75),
            run_deduplication=dedup.get('enabled', False),
            # QA correction
            qa_correction_enabled=qa_corr.get('enabled', True),
            qa_correction_max_attempts=qa_corr.get('max_attempts', 1),
            # Eval
            run_evaluation=eval_cfg.get('run_evaluation', False),
            # File reference
            config_file=path,
        )
    
    def save_yaml(self, path: str):
        """
        Save configuration to a YAML file.
        
        Args:
            path: Output YAML file path.
        """
        import yaml
        
        # Build nested YAML structure from flat config
        data = {
            'backend': {
                'active': self.backend.upper(),
                self.backend.lower(): {
                    k: v for k, v in {
                        'llm_model': self.llm_model,
                        'vlm_model': self.vlm_model,
                    }.items() if v is not None
                },
            },
            'paths': {
                'input_pdf_dir': self.input_dir,
                'output_dir': self.output_dir,
            },
            'qa_generation': {
                'num_qa_pairs': self.num_qa_pairs,
                'type': self.qa_type,
            },
            'retrieval': {
                'multihop': {
                    'max_depth': self.max_depth,
                    'max_breadth': self.max_breadth,
                    'chunks_per_search': self.chunks_per_search,
                    'chunk_addition_mode': self.chunk_addition_mode,
                }
            },
            'embedding': {
                'model': self.embedding_model,
                'batch_size': self.embed_batch_size,
                'cache_embeddings': self.cache_embeddings,
            },
            'parallel': {
                'num_workers': self.num_cpu_workers,
                'qa_max_workers': self.max_workers,
                'dedup_max_workers': self.dedup_max_workers,
            },
            'rate_limiting': {
                'requests_per_minute': self.requests_per_minute,
                'burst_size': self.burst_size,
            },
            'pdf_processing': {
                'image_resolution_scale': self.image_resolution_scale,
            },
        }
        
        if self.embedding_gpus:
            data['embedding']['gpus'] = self.embedding_gpus
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# =============================================================================
# Results
# =============================================================================
@dataclass
class MiRAGEResults:
    """
    Container for MiRAGE pipeline results.
    
    Supports iteration, indexing, save/load, and DataFrame conversion.
    
    Example::
    
        results = pipeline.run()
        
        # Iterate
        for qa in results:
            print(qa['question'])
        
        # Index
        first = results[0]
        
        # Length
        print(f"Generated {len(results)} QA pairs")
        
        # Save/Load
        results.save("dataset.json")
        loaded = MiRAGEResults.load("dataset.json")
        
        # pandas
        df = results.to_dataframe()
    """
    qa_pairs: List[Dict[str, Any]] = field(default_factory=list)
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    output_dir: Optional[str] = None
    config: Optional[MiRAGEConfig] = None
    
    def __len__(self) -> int:
        return len(self.qa_pairs)
    
    def __iter__(self):
        return iter(self.qa_pairs)
    
    def __getitem__(self, idx):
        return self.qa_pairs[idx]
    
    def __bool__(self) -> bool:
        return len(self.qa_pairs) > 0
    
    def __repr__(self) -> str:
        return (
            f"MiRAGEResults(qa_pairs={len(self.qa_pairs)}, "
            f"chunks={len(self.chunks)}, "
            f"output_dir='{self.output_dir}')"
        )
    
    @property
    def questions(self) -> List[str]:
        """List of all questions."""
        return [qa.get('question', '') for qa in self.qa_pairs]
    
    @property
    def answers(self) -> List[str]:
        """List of all answers."""
        return [qa.get('answer', '') for qa in self.qa_pairs]
    
    def save(self, path: Optional[str] = None, format: str = "json") -> str:
        """
        Save QA pairs to a file.
        
        Args:
            path: Output path. If None, saves to ``output_dir/qa_pairs.json``.
            format: ``"json"`` (default) or ``"jsonl"`` (one JSON object per line).
        
        Returns:
            Path to the saved file.
        """
        if path is None:
            path = os.path.join(self.output_dir or ".", "qa_pairs.json")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        if format == "jsonl":
            with open(path, 'w') as f:
                for qa in self.qa_pairs:
                    f.write(json.dumps(qa) + '\n')
        else:
            with open(path, 'w') as f:
                json.dump(self.qa_pairs, f, indent=2)
        
        logger.info(f"Saved {len(self.qa_pairs)} QA pairs to {path}")
        return path
    
    def to_dataframe(self):
        """
        Convert QA pairs to a pandas DataFrame.
        
        Returns:
            pandas.DataFrame with columns for question, answer, type, difficulty, etc.
        """
        import pandas as pd
        if not self.qa_pairs:
            return pd.DataFrame()
        
        rows = []
        for qa in self.qa_pairs:
            row = {
                'question': qa.get('question', ''),
                'answer': qa.get('answer', ''),
                'question_type': qa.get('question_type', ''),
                'difficulty': qa.get('difficulty', ''),
                'num_hops': qa.get('num_hops', 0),
            }
            for k, v in qa.items():
                if k not in row and isinstance(v, (str, int, float, bool)):
                    row[k] = v
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a dictionary."""
        return {
            'qa_pairs': self.qa_pairs,
            'stats': self.stats,
            'num_qa_pairs': len(self.qa_pairs),
            'num_chunks': len(self.chunks),
        }
    
    def filter(self, **kwargs) -> "MiRAGEResults":
        """
        Filter QA pairs by field values.
        
        Example::
        
            multihop = results.filter(question_type="multihop")
            hard = results.filter(difficulty="hard")
        
        Returns:
            New MiRAGEResults with filtered QA pairs.
        """
        filtered = []
        for qa in self.qa_pairs:
            if all(qa.get(k) == v for k, v in kwargs.items()):
                filtered.append(qa)
        return MiRAGEResults(
            qa_pairs=filtered,
            chunks=self.chunks,
            stats=self.stats,
            output_dir=self.output_dir,
            config=self.config,
        )
    
    def sample(self, n: int, seed: int = 42) -> "MiRAGEResults":
        """
        Random sample of QA pairs.
        
        Args:
            n: Number of QA pairs to sample.
            seed: Random seed for reproducibility.
        
        Returns:
            New MiRAGEResults with sampled QA pairs.
        """
        import random
        rng = random.Random(seed)
        sampled = rng.sample(self.qa_pairs, min(n, len(self.qa_pairs)))
        return MiRAGEResults(
            qa_pairs=sampled,
            chunks=self.chunks,
            stats=self.stats,
            output_dir=self.output_dir,
            config=self.config,
        )
    
    @classmethod
    def load(cls, path: str) -> "MiRAGEResults":
        """
        Load QA pairs from a JSON or JSONL file.
        
        Args:
            path: Path to JSON or JSONL file.
        
        Returns:
            MiRAGEResults instance.
        """
        path = str(path)
        
        if path.endswith('.jsonl'):
            qa_pairs = []
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        qa_pairs.append(json.loads(line))
            return cls(qa_pairs=qa_pairs)
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return cls(qa_pairs=data)
        elif isinstance(data, dict) and 'qa_pairs' in data:
            return cls(qa_pairs=data['qa_pairs'], stats=data.get('stats', {}))
        else:
            return cls(qa_pairs=[data] if isinstance(data, dict) else data)


# =============================================================================
# Main Pipeline Class
# =============================================================================
class MiRAGE:
    """
    MiRAGE: Multimodal Multihop RAG Evaluation Dataset Generator.
    
    A complete pipeline for generating QA datasets from documents.
    
    =========================================================================
    Quick Start:
    =========================================================================
    
    ::
    
        from mirage import MiRAGE
        
        pipeline = MiRAGE(
            input_dir="data/my_docs",
            output_dir="output/results",
            backend="gemini",
            api_key="your-gemini-api-key",
        )
        results = pipeline.run()
        results.save("my_dataset.json")
    
    =========================================================================
    From Config:
    =========================================================================
    
    ::
    
        pipeline = MiRAGE.from_config("config.yaml")
        pipeline.configure(num_qa_pairs=100)
        results = pipeline.run()
    
    =========================================================================
    Advanced:
    =========================================================================
    
    ::
    
        pipeline = MiRAGE(
            input_dir="data/papers",
            output_dir="output/papers_qa",
            backend="gemini",
            api_key="your-key",
            num_qa_pairs=200,
            max_depth=3,
            embedding_model="nomic",
            reranker_model="gemini_vlm",
            ocr_engine="easyocr",
            device="cuda:0",
            embedding_gpus=[0],
            run_deduplication=True,
        )
        results = pipeline.run()
        df = results.to_dataframe()
    """
    
    def __init__(self, config: Optional[MiRAGEConfig] = None, **kwargs):
        """
        Initialize MiRAGE pipeline.
        
        Can be initialized with a MiRAGEConfig object OR keyword arguments.
        
        Args:
            config: MiRAGEConfig instance (takes precedence over kwargs).
            **kwargs: Configuration parameters (see MiRAGEConfig for all options).
        
        Example::
        
            # With keyword arguments
            pipeline = MiRAGE(
                input_dir="docs",
                backend="gemini",
                api_key="key",
                num_qa_pairs=50,
            )
            
            # With config object
            config = MiRAGEConfig(backend="gemini", api_key="key")
            pipeline = MiRAGE(config=config)
        """
        if config is not None:
            self.config = config
        else:
            self.config = MiRAGEConfig(**kwargs)
        self._is_configured = False
    
    @classmethod
    def from_config(cls, config_path: str, **overrides) -> "MiRAGE":
        """
        Create pipeline from a YAML configuration file.
        
        Args:
            config_path: Path to config.yaml
            **overrides: Override any config values (e.g., num_qa_pairs=50)
        
        Returns:
            Configured MiRAGE pipeline.
        
        Example::
        
            pipeline = MiRAGE.from_config("config.yaml", num_qa_pairs=50, api_key="key")
        """
        config = MiRAGEConfig.from_yaml(config_path)
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        instance = cls.__new__(cls)
        instance.config = config
        instance._is_configured = False
        return instance
    
    def configure(self, **kwargs) -> "MiRAGE":
        """
        Update pipeline configuration. Returns self for method chaining.
        
        Args:
            **kwargs: Any MiRAGEConfig parameter to update.
        
        Returns:
            self (for chaining)
        
        Raises:
            ValueError: If an unknown parameter is passed.
        
        Example::
        
            results = pipeline.configure(num_qa_pairs=100, max_depth=3).run()
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                valid = list(self.config.__dataclass_fields__.keys())
                raise ValueError(
                    f"Unknown parameter: '{key}'. Valid parameters: {valid}"
                )
        self._is_configured = False
        return self
    
    def _setup_environment(self):
        """Apply MiRAGEConfig to environment variables before pipeline import."""
        cfg = self.config
        
        # Setup device
        from mirage.utils.device import setup_device_environment
        setup_device_environment()
        
        # Backend and API key
        os.environ["MIRAGE_BACKEND"] = cfg.backend.upper()
        if cfg.api_key:
            if cfg.backend.lower() == "gemini":
                os.environ["GEMINI_API_KEY"] = cfg.api_key
                os.environ["GOOGLE_API_KEY"] = cfg.api_key
            elif cfg.backend.lower() == "openai":
                os.environ["OPENAI_API_KEY"] = cfg.api_key
        
        # Models
        if cfg.llm_model:
            os.environ["MIRAGE_MODEL"] = cfg.llm_model
        if cfg.embedding_model and cfg.embedding_model != "auto":
            os.environ["MIRAGE_EMBEDDING_MODEL"] = cfg.embedding_model
        if cfg.reranker_model:
            os.environ["MIRAGE_RERANKER_MODEL"] = cfg.reranker_model
        
        # Paths
        os.environ["MIRAGE_INPUT_DIR"] = str(cfg.input_dir)
        os.environ["MIRAGE_OUTPUT_DIR"] = str(cfg.output_dir)
        
        # QA generation
        os.environ["MIRAGE_NUM_QA_PAIRS"] = str(cfg.num_qa_pairs)
        os.environ["MIRAGE_MAX_WORKERS"] = str(cfg.max_workers)
        os.environ["MIRAGE_MAX_DEPTH"] = str(cfg.max_depth)
        
        # Pipeline flags
        if cfg.skip_pdf_processing:
            os.environ["MIRAGE_SKIP_PDF_PROCESSING"] = "1"
        if cfg.skip_chunking:
            os.environ["MIRAGE_SKIP_CHUNKING"] = "1"
        if cfg.run_deduplication:
            os.environ["MIRAGE_RUN_DEDUPLICATION"] = "1"
        if cfg.run_evaluation:
            os.environ["MIRAGE_RUN_EVALUATION"] = "1"
        
        # Device
        if cfg.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        # Config file
        if cfg.config_file and os.path.exists(cfg.config_file):
            os.environ["MIRAGE_CONFIG"] = cfg.config_file
        
        self._is_configured = True
    
    def _get_project_root(self) -> Path:
        """Find project root (where config.yaml lives), or use cwd."""
        # Check common locations
        for candidate in [
            Path(__file__).parent.parent.parent,  # src/mirage/api.py -> project root
            Path.cwd(),
        ]:
            if (candidate / "config.yaml").exists():
                return candidate
        return Path.cwd()
    
    def preflight(self) -> bool:
        """
        Run preflight checks to validate the pipeline configuration.
        
        Tests all services: config, API key, LLM, VLM, embeddings, reranker.
        
        Returns:
            True if all checks passed.
        
        Example::
        
            if pipeline.preflight():
                results = pipeline.run()
            else:
                print("Fix configuration issues first")
        """
        self._setup_environment()
        
        original_cwd = os.getcwd()
        project_root = self._get_project_root()
        os.chdir(project_root)
        
        try:
            from mirage.utils.preflight import run_preflight_checks
            all_passed, _ = run_preflight_checks()
            return all_passed
        finally:
            os.chdir(original_cwd)
    
    def run(self) -> MiRAGEResults:
        """
        Run the complete MiRAGE pipeline.
        
        Executes all steps:
            1. Document conversion (PDF/HTML -> Markdown)
            2. Semantic chunking
            3. Embedding & FAISS indexing
            4. Multi-hop QA generation
            5. (Optional) Deduplication
            6. (Optional) Evaluation
        
        Returns:
            MiRAGEResults with generated QA pairs, chunks, and statistics.
        
        Example::
        
            results = pipeline.run()
            print(f"Generated {len(results)} QA pairs")
            results.save("dataset.json")
        """
        if not self._is_configured:
            self._setup_environment()
        
        cfg = self.config
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        
        original_cwd = os.getcwd()
        project_root = self._get_project_root()
        os.chdir(project_root)
        
        try:
            from mirage.main import run_pipeline
            
            run_pipeline()
            
            return self._collect_results()
        finally:
            os.chdir(original_cwd)
    
    def _collect_results(self) -> MiRAGEResults:
        """Collect results from the output directory after pipeline execution."""
        cfg = self.config
        out = Path(cfg.output_dir)
        
        # Load QA pairs
        qa_pairs = []
        for qa_file in ["qa_multihop_pass.json", "qa_deduplicated.json"]:
            p = out / qa_file
            if p.exists():
                with open(p, 'r') as f:
                    qa_pairs = json.load(f)
                break
        
        # Load chunks
        chunks = []
        chunks_file = out / "chunks.json"
        if chunks_file.exists():
            with open(chunks_file, 'r') as f:
                chunks = json.load(f)
        
        # Gather statistics
        token_stats = {}
        try:
            from mirage.core.llm import get_token_stats
            token_stats = get_token_stats()
        except Exception:
            pass
        
        stats = {
            "total_qa_pairs": len(qa_pairs),
            "total_chunks": len(chunks),
            "backend": cfg.backend,
            "num_qa_pairs_requested": cfg.num_qa_pairs,
            **token_stats,
        }
        
        return MiRAGEResults(
            qa_pairs=qa_pairs,
            chunks=chunks,
            stats=stats,
            output_dir=cfg.output_dir,
            config=cfg,
        )
    
    def get_config(self) -> MiRAGEConfig:
        """Return the current pipeline configuration."""
        return self.config
    
    def save_config(self, path: str):
        """
        Save the current configuration to a YAML file.
        
        Args:
            path: Output YAML file path.
        
        Example::
        
            pipeline.save_config("my_config.yaml")
        """
        self.config.save_yaml(path)
    
    def __repr__(self) -> str:
        c = self.config
        parts = [f"backend='{c.backend}'", f"input_dir='{c.input_dir}'"]
        if c.num_qa_pairs:
            parts.append(f"num_qa_pairs={c.num_qa_pairs}")
        if c.device:
            parts.append(f"device='{c.device}'")
        if c.embedding_model != "auto":
            parts.append(f"embedding_model='{c.embedding_model}'")
        return f"MiRAGE({', '.join(parts)})"
