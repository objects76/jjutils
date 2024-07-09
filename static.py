
import inspect

# dynamic alloc
def get_local_static(name, default, obj = None):
    if obj is None:
        caller_frame = inspect.currentframe().f_back
        code = caller_frame.f_code
        obj = caller_frame.f_globals[code.co_name]

    if not hasattr(obj, name):
        setattr(obj, name, default)
    return getattr(obj, name)


# decorator
def static_vars(**kwargs):
    def decorate(func):
        for k, v in kwargs.items():
            setattr(func, k, v)
        return func
    return decorate