"""
Utility decorators for Lambda functions.
"""
import functools
from datetime import datetime

from common.logging import get_custom_logger

logger = get_custom_logger("blackbox.common.decorators")


def measure_execution_time(function_name=None):
    """
    Decorator to measure and log execution time of a function.

    Args:
        function_name (str, optional): Name to use in the log message.
            If None, the function's __name__ will be used.

    Example usage:
        @measure_execution_time()
        def my_function():
            # function code

        @measure_execution_time("Custom Function Name")
        def another_function():
            # function code
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the name to use in the log message
            name = function_name if function_name else func.__name__
            start_time = datetime.now()
            result = func(*args, **kwargs)
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"{name} completed in {processing_time:.2f} seconds")
            return result

        return wrapper

    # Handle both @measure_execution_time and @measure_execution_time()
    if callable(function_name):
        func = function_name
        function_name = None
        return decorator(func)

    return decorator
