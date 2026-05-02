from functools import wraps
from logging import getLogger
from typing import Any, Callable

_LOGGER = getLogger(__name__)


def safe_run(default_return_value: Any, raise_anyway: bool = False) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                _LOGGER.error(f"Error occurred while executing {func.__name__}: {e}")
                if raise_anyway:
                    raise e
                return default_return_value

        return wrapper

    return decorator
