from functools import wraps
import time
from datetime import datetime
import json

def stats(name, func_name, value, exec_time = str(datetime.now())): 
    return {
        "decorator_name": name,
        "function_name": func_name,
        "value": value,
        "exec_time": exec_time
    }

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        with open("/shared/project/CourseQA-RAG/log/log.jsonl", "a") as f:
            timer_stats = stats("timer", func.__name__, f"{(end - start):.5f}")
            json.dump(timer_stats, f)
            f.write("\n")
        return result
    return wrapper