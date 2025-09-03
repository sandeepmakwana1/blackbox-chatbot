import copy
import logging
import os
import sys
import time
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from elasticapm import Client
from elasticapm.handlers.logging import LoggingHandler as ElasticLogHandler

from common.config import get_elastic_config

request_id_context: ContextVar[str] = ContextVar("request_id", default="no-request-id")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_context.get()
        return True


_LOG_CONFIGURED: bool = False
_LOG_FILE_PATH: Optional[Path] = None


class CustomElasticHandler(ElasticLogHandler):
    def emit(self, record):
        # Add request_id
        record.request_id = request_id_context.get()

        # Create formatted message
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        original_msg = record.getMessage()
        formatted_msg = f"{timestamp} - {record.levelname} - [{record.request_id} - {record.name}] - {original_msg}"
        if (
            record.name == "__main__"
        ):  # TODO : will modify later in the time of optimization
            formatted_msg = f"{timestamp} - {record.levelname} - [{record.request_id}] - {original_msg}"

        # Create a copy of record with formatted message
        elastic_record = copy.copy(record)
        elastic_record.msg = formatted_msg
        elastic_record.args = ()

        # Send to Elastic
        super().emit(elastic_record)


def _setup_logging(
    *,
    console: bool = True,
    file: bool = False,
    elastic: bool = True,
    log_level: str = "INFO",
    log_dir: str | Path = "./logs",
) -> Optional[Path]:
    """
    Configure root logging exactly once.
    Parameters:
        console, file, elastic : bool
            Enable/disable individual handlers.
        log_level : str
            Root log level (DEBUG/INFO/…)
        log_dir : str | Path
            Directory that will hold the rotating file (if `file=True`).

    Returns :
        Path | None
            Absolute path to the log file if `file=True`, else ``None``.
    """
    # root logger
    root = logging.getLogger()
    root.setLevel(log_level.upper())
    root.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(request_id)s - %(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Elastic
    if elastic:
        try:
            config = get_elastic_config()
            apm_client = Client(config)
            eh = CustomElasticHandler(client=apm_client)
            eh.setLevel(log_level.upper())
            root.addHandler(eh)
            print("Elastic APM logging: enabled")
        except Exception:
            root.warning(
                "APM HANDLER NOT WORKING THIS COULD BE AN ISSUE if not running on local"
            )

    # File
    log_path: Optional[Path] = None
    if file:
        log_path = Path(log_dir).resolve()
        log_path.mkdir(parents=True, exist_ok=True)
        log_path = log_path / "logfilt.log"

        fh = RotatingFileHandler(
            log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        fh.setFormatter(formatter)
        fh.setLevel(log_level.upper())
        fh.addFilter(RequestIdFilter())
        root.addHandler(fh)

        print(f"File logging: enabled → {log_path}")

    # Console
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(log_level.upper())
        ch.addFilter(RequestIdFilter())
        root.addHandler(ch)

        print("Console logging: enabled")

    root.propagate = False
    return log_path


def get_custom_logger(
    name: str,
    *,
    console: bool = True,
    file: bool = False,
    elastic: bool = True,
    log_level: str = "",
) -> logging.Logger:
    """
    Return a logger for the caller and *optionally* bootstrap root logging.

    Parameters :
        name : str
            Usually ``__name__``.
        console, file : bool
            Enable console/file handlers. Defaults are ``True`` for backward compatibility.
        elastic : bool | None
            • ``True`` → always attempt Elastic handler
            • ``False`` → skip Elastic handler
            `ELASTIC_ENABLED=true|false`.
        log_level : str | None
            Overrides the environment variable `LOG_LEVEL` **only** on the very
            first call (subsequent calls cannot reconfigure the root logger).

    Usage :
        >>> logger = get_custom_logger(__name__, console=True, file=False, elastic=True)
        >>> logger.info("Hello, world!")
    """
    global _LOG_CONFIGURED, _LOG_FILE_PATH

    if not _LOG_CONFIGURED:
        _LOG_FILE_PATH = _setup_logging(
            console=console,
            file=file,
            elastic=elastic,
            log_level=(log_level or os.getenv("LOG_LEVEL", "INFO")),
        )
        _LOG_CONFIGURED = True

    return logging.getLogger(name)


# Convenience helpers retained from original ---------------------------------
def set_request_id(request_id: str):  # noqa: D401
    """Set the `request_id` for the current async context."""
    request_id_context.set(request_id or "no-request-id")


def get_log_file_path() -> Optional[Path]:
    """Return the path to the rotating file handler, if one exists."""
    return _LOG_FILE_PATH
