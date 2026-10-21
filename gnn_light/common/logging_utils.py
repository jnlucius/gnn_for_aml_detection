import json
import logging
import os
from datetime import datetime, timezone


def get_logger(name: str, size: str = "", run_tag: str = "", log_dir: str = "./logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = os.path.join(log_dir, f"{name}_{size}{run_tag}_{timestamp}.log")

    logger = logging.getLogger(f"gnn_light.{name}{run_tag}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.log_path = log_path  # pyright: ignore
    logger.info(f"Logging to {log_path}")
    return logger


def log_json(logger: logging.Logger, event: str, payload: dict) -> None:
    record = {"event": event, "timestamp": datetime.now(timezone.utc).isoformat(), **payload}
    logger.info("RESULT_JSON " + json.dumps(record, default=str))
