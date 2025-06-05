from typing import Any
import diskcache
import time
import contextlib

@contextlib.contextmanager
def running_time(description: str = "Exec"):
    tick = time.time()
    print(f"{description}: start...")
    try:
        yield
    finally:
        print(f"{description}: {time.time() - tick:.2f} seconds")


class DiskCache:
    def __init__(self, task:str, base_dir="./cache") -> None:
        cache_dir = str(Path(base_dir) / task)
        self._cache = diskcache.Cache(cache_dir, cull_limit=0)

    def get(self, *args, **kwargs) -> Any:
        tokens = [str(arg) for arg in args]
        tokens.extend([f"{key}={value!s}" for key, value in kwargs.items()])
        self.key = "~".join(tokens)
        return self._cache.get(self.key, None)

    def set(self, value:Any):
        self._cache[self.key] = value

    def keys(self) -> list:

        return [k for k in self._cache.iterkeys()]


