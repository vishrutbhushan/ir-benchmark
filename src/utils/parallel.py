"""Ordered worker mapping with bounded inputs on Python 3.12 and newer."""
from itertools import islice


def bounded_map(executor, function, tasks, window, chunksize=1):
    tasks = iter(tasks)
    while batch := list(islice(tasks, window)):
        yield from executor.map(function, batch, chunksize=chunksize)
