from importlib import import_module
import inspect
import os

def get_interruption_policy_callables():
    """Return dictionary with all callables defined in files in this directory.

    This file is excluded from the search."""
    this_file = os.path.split(__file__)[-1]
    directory = os.path.dirname(__file__)
    exclude = [this_file]
    ip_files = [f for f in os.listdir(directory) 
            if f.endswith(".py") and f not in exclude]
    ip_callables = {}
    for f in ip_files:
        path = os.path.join(directory, f)
        relative_import_string = "." + inspect.getmodulename(path)
        module = import_module(relative_import_string, package=__package__)
        for name in dir(module):
            obj = getattr(module, name)
            if callable(obj):
                ip_callables[name] = obj

    return ip_callables


def get_ip_callable(name):
    return get_interruption_policy_callables()[name]
