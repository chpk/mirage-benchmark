"""
MiRAGE High-Level Python API

Provides a clean, library-style interface for MiRAGE - similar to how
HuggingFace Transformers, OpenAI, or scikit-learn expose their APIs.

Usage:
    from mirage import MiRAGE

    # Quick start - minimal configuration
    pipeline = MiRAGE(
        input_dir="data/my_documents",
        output_dir="output/results",
        backend="gemini",
        api_key="your-api-key",
    )
    results = pipeline.run()
    results.save("my_dataset.json")

    # Advanced - full control
    pipeline = MiRAGE.from_config("config.yaml")
    pipeline.configure(
        num_qa_pairs=100,
        max_depth=3,
        embedding_model="nomic",
        device="cuda:0",
    )
    results = pipeline.run()
"""

import os
import json
import tempfile
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
    Configuration for the MiRAGE pipeline.
    
    All parameters have sensible defaults. At minimum, you need:
    - input_dir: Path to documents (PDFs, HTML, etc.)
    - output_dir: Where to save results
    - backend: LLM backend ("gemini", "openai", "ollama")
    - api_key: API key for the chosen backend
    
    Example:
        config = MiRAGEConfig(
            input_dir="data/docs",
            output_dir="output",
            backend="gemini",
            api_key="your-key",
            num_qa_pairs=50,
        )
    """
    # === Required ===
    input_dir: str = "data/documents"
    output_dir: str = "output/results"
    backend: str = "gemini"
    api_key: Optional[str] = None
    
    # === Model Selection ===
    llm_model: Optional[str] = None         # Auto-selected per backend
    vlm_model: Optional[str] = None         # Auto-selected per backend
    embedding_model: str = "auto"           # "auto", "nomic", "bge_m3", "gemini", "qwen3_vl"
    reranker_model: str = "gemini_vlm"      # "gemini_vlm", "monovlm", "text_embedding"
    
    # === QA Generation ===
    num_qa_pairs: int = 10                  # Target number of QA pairs
    qa_type: str = "multihop"               # "multihop", "multimodal", "text", "mix"
    max_depth: int = 2                      # Multi-hop context retrieval depth
    max_breadth: int = 5                    # Search breadth per verification
    
    # === Processing ===
    max_workers: int = 4                    # Parallel workers for QA generation
    num_cpu_workers: int = 3                # CPU workers for document processing
    embed_batch_size: int = 16              # Embedding batch size
    max_pages: Optional[int] = None         # Limit input pages (None = no limit)
    
    # === Device ===
    device: Optional[str] = None            # "cuda", "cuda:0", "cpu", None (auto-detect)
    embedding_gpus: Optional[List[int]] = None  # GPU IDs for embeddings
    
    # === Pipeline Control ===
    skip_pdf_processing: bool = False       # Skip PDF-to-markdown conversion
    skip_chunking: bool = False             # Skip semantic chunking
    run_deduplication: bool = False          # Enable QA deduplication
    run_evaluation: bool = False             # Enable evaluation metrics
    enable_checkpointing: bool = True       # Resume from interruptions
    
    # === Advanced ===
    config_file: Optional[str] = None       # Path to config.yaml for full control
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MiRAGEConfig":
        """Create config from dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)
    
    @classmethod
    def from_yaml(cls, path: str) -> "MiRAGEConfig":
        """Load config from a YAML file."""
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
        
        return cls(
            input_dir=paths.get('input_pdf_dir', 'data/documents'),
            output_dir=paths.get('output_dir', 'output/results'),
            backend=active,
            llm_model=active_cfg.get('llm_model'),
            vlm_model=active_cfg.get('vlm_model'),
            embedding_model=embedding.get('model', 'auto'),
            num_qa_pairs=qa_gen.get('num_qa_pairs', 10),
            qa_type=qa_gen.get('type', 'multihop'),
            max_depth=retrieval.get('max_depth', 2),
            max_breadth=retrieval.get('max_breadth', 5),
            max_workers=parallel.get('qa_max_workers', 4),
            num_cpu_workers=parallel.get('num_workers', 3),
            embed_batch_size=embedding.get('batch_size', 16),
            embedding_gpus=embedding.get('gpus'),
            config_file=path,
        )


# =============================================================================
# Results
# =============================================================================
@dataclass
class MiRAGEResults:
    """
    Container for pipeline results.
    
    Attributes:
        qa_pairs: List of generated QA pairs
        chunks: List of semantic chunks from documents
        stats: Pipeline statistics (tokens, timing, etc.)
        output_dir: Directory where full results are saved
        config: The configuration used for this run
    
    Example:
        results = pipeline.run()
        print(f"Generated {len(results)} QA pairs")
        
        for qa in results:
            print(qa['question'])
        
        results.save("my_dataset.json")
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
    
    def __repr__(self) -> str:
        return (
            f"MiRAGEResults(qa_pairs={len(self.qa_pairs)}, "
            f"chunks={len(self.chunks)}, "
            f"output_dir='{self.output_dir}')"
        )
    
    def save(self, path: Optional[str] = None, format: str = "json") -> str:
        """
        Save QA pairs to a file.
        
        Args:
            path: Output file path. If None, saves to output_dir/qa_pairs.json
            format: "json" or "jsonl"
        
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
        """Convert QA pairs to a pandas DataFrame."""
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
            # Add any extra fields
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
    
    @classmethod
    def load(cls, path: str) -> "MiRAGEResults":
        """Load QA pairs from a JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Handle both raw list and wrapped dict formats
        if isinstance(data, list):
            return cls(qa_pairs=data)
        elif isinstance(data, dict) and 'qa_pairs' in data:
            return cls(
                qa_pairs=data['qa_pairs'],
                stats=data.get('stats', {}),
            )
        else:
            return cls(qa_pairs=data if isinstance(data, list) else [data])


# =============================================================================
# Main Pipeline Class
# =============================================================================
class MiRAGE:
    """
    MiRAGE: Multimodal Multihop RAG Evaluation Dataset Generator
    
    High-level API for generating QA datasets from documents.
    
    Quick Start:
        >>> from mirage import MiRAGE
        >>> pipeline = MiRAGE(
        ...     input_dir="data/my_docs",
        ...     output_dir="output/results",
        ...     backend="gemini",
        ...     api_key="your-gemini-api-key",
        ... )
        >>> results = pipeline.run()
        >>> print(f"Generated {len(results)} QA pairs")
        >>> results.save("my_dataset.json")
    
    From Config File:
        >>> pipeline = MiRAGE.from_config("config.yaml")
        >>> pipeline.configure(num_qa_pairs=100)
        >>> results = pipeline.run()
    
    Advanced Usage:
        >>> pipeline = MiRAGE(
        ...     input_dir="data/papers",
        ...     output_dir="output/papers_qa",
        ...     backend="gemini",
        ...     api_key="your-key",
        ...     num_qa_pairs=200,
        ...     max_depth=3,
        ...     embedding_model="nomic",
        ...     device="cuda:0",
        ...     run_deduplication=True,
        ... )
        >>> results = pipeline.run()
        >>> df = results.to_dataframe()
        >>> df.to_csv("qa_dataset.csv")
    """
    
    def __init__(self, **kwargs):
        """
        Initialize MiRAGE pipeline.
        
        Args:
            input_dir: Path to directory containing documents (PDF, HTML, etc.)
            output_dir: Path to output directory for results
            backend: LLM backend - "gemini", "openai", or "ollama"
            api_key: API key for the chosen backend
            llm_model: LLM model name (auto-selected if None)
            vlm_model: VLM model name (auto-selected if None)
            embedding_model: Embedding model - "auto", "nomic", "bge_m3", "gemini"
            reranker_model: Reranker model - "gemini_vlm", "monovlm", "text_embedding"
            num_qa_pairs: Target number of QA pairs to generate
            qa_type: QA type - "multihop", "multimodal", "text", "mix"
            max_depth: Maximum depth for multi-hop context retrieval
            max_breadth: Maximum breadth for search verification
            max_workers: Number of parallel workers
            embed_batch_size: Batch size for embeddings
            device: Device - "cuda", "cuda:0", "cpu", or None (auto-detect)
            embedding_gpus: List of GPU IDs for embedding model
            skip_pdf_processing: Skip PDF-to-markdown conversion
            skip_chunking: Skip semantic chunking
            run_deduplication: Enable QA deduplication
            run_evaluation: Enable evaluation metrics
            config_file: Path to config.yaml for full configuration
        """
        self.config = MiRAGEConfig(**kwargs)
        self._is_configured = False
    
    @classmethod
    def from_config(cls, config_path: str, **overrides) -> "MiRAGE":
        """
        Create a MiRAGE pipeline from a YAML configuration file.
        
        Args:
            config_path: Path to config.yaml
            **overrides: Override any config values
        
        Returns:
            Configured MiRAGE pipeline instance.
        
        Example:
            >>> pipeline = MiRAGE.from_config("config.yaml", num_qa_pairs=50)
        """
        config = MiRAGEConfig.from_yaml(config_path)
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        instance = cls.__new__(cls)
        instance.config = config
        instance._is_configured = False
        return instance
    
    def configure(self, **kwargs) -> "MiRAGE":
        """
        Update pipeline configuration.
        
        Returns self for method chaining.
        
        Example:
            >>> pipeline.configure(num_qa_pairs=100, max_depth=3).run()
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(
                    f"Unknown configuration parameter: '{key}'. "
                    f"Valid parameters: {list(self.config.__dataclass_fields__.keys())}"
                )
        self._is_configured = False
        return self
    
    def _setup_environment(self):
        """Configure environment variables from MiRAGEConfig before pipeline import."""
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
        
        # Model selection
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
        if cfg.device:
            if cfg.device == "cpu":
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        # Config file
        if cfg.config_file and os.path.exists(cfg.config_file):
            os.environ["MIRAGE_CONFIG"] = cfg.config_file
        
        self._is_configured = True
    
    def preflight(self) -> bool:
        """
        Run preflight checks to validate the pipeline configuration.
        
        Returns:
            True if all checks passed, False otherwise.
        
        Example:
            >>> pipeline = MiRAGE(backend="gemini", api_key="your-key")
            >>> if pipeline.preflight():
            ...     results = pipeline.run()
        """
        self._setup_environment()
        
        # Change to project root if config.yaml exists there
        original_cwd = os.getcwd()
        project_root = Path(__file__).parent.parent.parent
        if (project_root / "config.yaml").exists():
            os.chdir(project_root)
        
        try:
            from mirage.utils.preflight import run_preflight_checks
            all_passed, _ = run_preflight_checks()
            return all_passed
        finally:
            os.chdir(original_cwd)
    
    def run(self) -> MiRAGEResults:
        """
        Run the full MiRAGE pipeline.
        
        Executes:
            1. Document conversion (PDF/HTML -> Markdown)
            2. Semantic chunking
            3. Embedding & indexing
            4. Multi-hop QA generation
            5. (Optional) Deduplication
            6. (Optional) Evaluation
        
        Returns:
            MiRAGEResults with generated QA pairs, chunks, and statistics.
        
        Example:
            >>> results = pipeline.run()
            >>> print(f"Generated {len(results)} QA pairs")
            >>> for qa in results:
            ...     print(qa['question'])
        """
        if not self._is_configured:
            self._setup_environment()
        
        cfg = self.config
        
        # Ensure output directory exists
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Change to project root if config.yaml exists there
        original_cwd = os.getcwd()
        project_root = Path(__file__).parent.parent.parent
        if (project_root / "config.yaml").exists():
            os.chdir(project_root)
        
        try:
            # Import and run the pipeline
            from mirage.main import run_pipeline
            from mirage.core.llm import get_token_stats
            
            run_pipeline()
            
            # Collect results
            qa_pairs = []
            qa_file = Path(cfg.output_dir) / "qa_multihop_pass.json"
            if qa_file.exists():
                with open(qa_file, 'r') as f:
                    qa_pairs = json.load(f)
            
            chunks = []
            chunks_file = Path(cfg.output_dir) / "chunks.json"
            if chunks_file.exists():
                with open(chunks_file, 'r') as f:
                    chunks = json.load(f)
            
            # Gather statistics
            token_stats = {}
            try:
                token_stats = get_token_stats()
            except:
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
        
        finally:
            os.chdir(original_cwd)
    
    def __repr__(self) -> str:
        return (
            f"MiRAGE(backend='{self.config.backend}', "
            f"input_dir='{self.config.input_dir}', "
            f"num_qa_pairs={self.config.num_qa_pairs})"
        )
