import psutil
import os
import time
from functools import wraps
from typing import Callable, Any

def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # in MB

def track_stats(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        start_memory = get_memory_usage()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = get_memory_usage()
        
        execution_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        print(f"Stats for {func.__name__}:")
        print(f"  Execution time: {execution_time:.2f} seconds")
        print(f"  Memory usage: {memory_used:.2f} MB")
        
        return result
    return wrapper