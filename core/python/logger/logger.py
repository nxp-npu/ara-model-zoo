# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import logging
import sys
from typing import ClassVar, Optional

import colorama
from colorama import Fore, Style

# Initialize colorama for cross-platform colored output
colorama.init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    # Color mapping for different log levels
    COLORS: ClassVar[dict[int, str]] = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA + Style.BRIGHT,
    }

    # Nicer level names
    LEVEL_NAMES: ClassVar[dict[int, str]] = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO ",
        logging.WARNING: "WARN ",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "CRIT ",
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelno, Fore.WHITE)
        level_name = self.LEVEL_NAMES.get(record.levelno, record.levelname)
        colored_levelname = f"{log_color}{level_name}{Style.RESET_ALL}"
        original_levelname = record.levelname
        record.levelname = colored_levelname
        formatted_message = super().format(record)
        record.levelname = original_levelname
        return formatted_message


def setup_logger(
    name: str = "Ara-model-zoo",
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    colored: bool = True,
) -> logging.Logger:
    """
    Set up a project-wide logger for console output with optional coloring.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        colored: Whether to use colored output (default: True)

    Returns:
        Configured logger instance
    """

    # Cleaner default format
    if format_string is None:
        format_string = "%(asctime)s %(levelname)s %(name)s → %(message)s"

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid adding multiple handlers if logger already exists
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Create formatter (colored or regular)
        if colored:
            formatter = ColoredFormatter(format_string, datefmt="%H:%M:%S")
        else:
            formatter = logging.Formatter(format_string, datefmt="%H:%M:%S")

        console_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(console_handler)

    return logger


def setup_minimal_logger(
    name: str = "Ara-model-zoo",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Set up a logger with minimal, clean formatting.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger with minimal formatting
    """
    format_string = "%(levelname)s %(message)s"
    return setup_logger(name, level, format_string, colored=True)


def setup_detailed_logger(
    name: str = "Ara-model-zoo",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Set up a logger with more detailed but clean formatting.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger with detailed formatting
    """
    format_string = (
        f"{Fore.BLUE}%(asctime)s{Style.RESET_ALL} "
        f"%(levelname)s "
        f"{Fore.CYAN}%(funcName)s(){Style.RESET_ALL} "
        f"-> %(message)s"
    )
    return setup_logger(name, level, format_string, colored=True)


def setup_compact_logger(
    name: str = "Ara-model-zoo",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Set up a logger with compact formatting.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger with compact formatting
    """
    format_string = "[%(asctime)s] %(levelname)s %(message)s"
    return setup_logger(name, level, format_string, colored=True)


# Create default project logger with clean format
project_logger = setup_detailed_logger()
