# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Logging utilities when working with pipecat pipelines."""

import asyncio
import sys

from loguru import logger


async def logger_context(coro, **kwargs):
    """Wrapper coroutine to contextualize the logger during the execution of `coro`."""
    with logger.contextualize(**kwargs):
        return await coro


def setup_default_logging(
    level: str = "INFO",
    stream_id: str | None = None,
    exclude_warning: bool = False,
):
    """Setup logger.

    Updates the logging format to include stream_id if available.

    Args:
        stream_id (str, optional) : Set the stream_id globally to use in log output.
            If you want different stream ids for different pipelines running
            in the same process use `logger_context` instead
        level (str): Logging level
        exclude_warning (bool): If True, exclude WARNING level logs while keeping DEBUG and INFO
    """
    logger_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "streamId={extra[stream_id]} - <level>{message}</level>"
    )
    logger.configure(extra={"stream_id": stream_id or "n/a"})  # Default values
    logger.remove()

    # Custom filter to exclude WARNING logs if requested
    def filter_warnings(record):
        if exclude_warning:
            return record["level"].name != "WARNING"
        return True

    logger.add(sys.stderr, format=logger_format, level=level, filter=filter_warnings)


def log_execution(func):
    """Decorator to log the start and end of an async function execution.

    Args:
        func (coroutine): The async function to be decorated.

    Returns:
        coroutine: The wrapped async function.
    """

    async def wrapper(*args, **kwargs):
        class_name = args[0].__class__.__name__ if args and hasattr(args[0], "__class__") else None
        name = f"{class_name}.{func.__name__}" if class_name else func.__name__
        args_str = ", ".join(str(arg)[:20] for arg in args)
        kwargs_str = ", ".join(f"{k}={str(v)[:20]}" for k, v in kwargs.items())
        parameters = f"{args_str}, {kwargs_str}" if kwargs else args_str

        logger.debug(f"Starting execution of {name}({parameters})")
        try:
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=1.0)
        except TimeoutError:
            logger.error(f"Finished execution of {name} with timeout after 1 second")
            return None

        logger.debug(f"Finished execution of {name}({parameters})")
        return result

    return wrapper
