import time, functools

def time_func(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        value = func(*args, **kwargs)
        t = time.perf_counter()-t0
        print(func.__name__, t)
        return value
    return wrapper
