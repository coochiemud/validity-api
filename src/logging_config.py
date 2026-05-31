"""
Validity — Logging
logging_config.py

Structured logging for the Validity pipeline.
Replaces ad-hoc print statements with levelled, formatted log output.

Usage:
    from logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Stage 1 complete", extra={"claims": 7})

Log levels:
    DEBUG   — detailed pipeline state (AST nodes, predicate maps)
    INFO    — stage progress, claim counts, verdicts
    WARNING — retries, low confidence claims, outside fragment
    ERROR   — API failures, parse errors, validation failures
    CRITICAL — pipeline crash, unrecoverable state
"""

from __future__ import annotations
import logging
import os
import sys
from typing import Optional


# ── Log format ────────────────────────────────────────────────────────────────

CONSOLE_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)-28s  %(message)s"
FILE_FORMAT    = "%(asctime)s  %(levelname)-8s  %(name)-28s  %(message)s"
DATE_FORMAT    = "%H:%M:%S"


# ── Validity logger names ─────────────────────────────────────────────────────

LOGGERS = {
    "validity.pipeline":    "Pipeline",
    "validity.extractor":   "Stage1",
    "validity.registry":    "Registry",
    "validity.translator":  "Stage2",
    "validity.validator":   "Validator",
    "validity.encoder":     "Encoder",
    "validity.solver":      "Solver",
    "validity.mapper":      "Mapper",
    "validity.renderer":    "Renderer",
    "validity.loader":      "Loader",
    "validity.resilience":  "Resilience",
}


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup_logging(
    level:    str = "INFO",
    log_file: Optional[str] = None,
    quiet:    bool = False,
) -> None:
    """
    Configure logging for the Validity pipeline.

    level:    Log level — DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_file: Optional path to write logs to file
    quiet:    If True, suppress console output (useful when piping output)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Root validity logger
    root = logging.getLogger("validity")
    root.setLevel(log_level)
    root.handlers.clear()

    # Console handler
    if not quiet:
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(log_level)
        console.setFormatter(logging.Formatter(CONSOLE_FORMAT, datefmt=DATE_FORMAT))
        root.addHandler(console)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Always write DEBUG to file
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
        root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a named Validity logger.
    Automatically prefixes with 'validity.' if not already prefixed.
    """
    if not name.startswith("validity."):
        name = f"validity.{name}"
    return logging.getLogger(name)


# ── Module-level loggers ──────────────────────────────────────────────────────
# Pre-defined loggers for each pipeline component.
# Import directly: from logging_config import pipeline_logger

pipeline_logger   = get_logger("validity.pipeline")
extractor_logger  = get_logger("validity.extractor")
registry_logger   = get_logger("validity.registry")
translator_logger = get_logger("validity.translator")
validator_logger  = get_logger("validity.validator")
encoder_logger    = get_logger("validity.encoder")
solver_logger     = get_logger("validity.solver")
mapper_logger     = get_logger("validity.mapper")
renderer_logger   = get_logger("validity.renderer")
loader_logger     = get_logger("validity.loader")
