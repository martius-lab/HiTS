from importlib import import_module
import inspect
import os

from .subtask_spec_factory import SubtaskSpecFactory

def get_subtask_spec_factory_classes():
    """Return dictionary with all factory classes defined in files in this directory.

    This file is excluded from the search."""
    this_file = os.path.split(__file__)[-1]
    directory = os.path.dirname(__file__)
    exclude = [this_file, "subtask_spec_factory.py"]
    factory_files = [f for f in os.listdir(directory) 
            if f.endswith(".py") and f not in exclude]
    factory_classes = {}
    for f in factory_files:
        path = os.path.join(directory, f)
        relative_import_string = "." + inspect.getmodulename(path)
        module = import_module(relative_import_string, package=__package__)
        for name in dir(module):
            obj = getattr(module, name)
            if inspect.isclass(obj):
                if issubclass(obj, SubtaskSpecFactory):
                    factory_classes[name] = obj

    return factory_classes


def get_subtask_spec_factory_class(name):
    """Get subtask spec factory class by name."""
    return get_subtask_spec_factory_classes()[name]
