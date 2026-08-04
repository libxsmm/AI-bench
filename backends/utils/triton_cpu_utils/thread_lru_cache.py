import functools
import threading


def thread_lru_cache(maxsize=128, typed=False):
    local = threading.local()

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                cached_func = local.cached_func
            except AttributeError:
                cached_func = functools.lru_cache(maxsize=maxsize, typed=typed)(func)
                local.cached_func = cached_func
            return cached_func(*args, **kwargs)

        return wrapper

    return decorator
