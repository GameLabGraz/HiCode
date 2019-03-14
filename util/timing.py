import time


def time_f(func):
    def f(*args, **kwargs):
        before = time.time()
        rv = func(*args, **kwargs)
        after = time.time()
        print("Function {0} took {1}s".format(func.__name__, after - before), )
        return rv
    return f
