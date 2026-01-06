"""
Core module for MiRAGE - LLM/VLM interfaces, prompts, and configuration.
"""

from mirage.core.llm import (
    call_llm_simple,
    call_vlm_interweaved,
    call_vlm_with_multiple_images,
    batch_call_vlm_interweaved,
    setup_logging,
    test_llm_connection,
    test_vlm_connection,
    BACKEND,
    LLM_MODEL_NAME,
    VLM_MODEL_NAME,
    GEMINI_RPM,
    GEMINI_BURST,
)

from mirage.core.prompts import PROMPTS, PROMPTS_CHUNK

from mirage.core.config import load_config, get_config_value, ConfigLoader

__all__ = [
    # LLM functions
    "call_llm_simple",
    "call_vlm_interweaved",
    "call_vlm_with_multiple_images",
    "batch_call_vlm_interweaved",
    "setup_logging",
    "test_llm_connection",
    "test_vlm_connection",
    "BACKEND",
    "LLM_MODEL_NAME",
    "VLM_MODEL_NAME",
    "GEMINI_RPM",
    "GEMINI_BURST",
    # Prompts
    "PROMPTS",
    "PROMPTS_CHUNK",
    # Config
    "load_config",
    "get_config_value",
    "ConfigLoader",
]
