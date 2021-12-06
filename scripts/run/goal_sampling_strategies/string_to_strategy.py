from importlib import import_module
import inspect
import os

from graph_rl.algorithms import GoalSamplingStrategy

def get_goal_sampling_strategy_classes():
    """Return dictionary with all goal sampling strategy classes defined in files in this directory.

    This file is excluded from the search."""

    this_file = os.path.split(__file__)[-1]
    directory = os.path.dirname(__file__)
    exclude = [this_file]
    factory_files = [f for f in os.listdir(directory) 
            if f.endswith(".py") and f not in exclude]
    gss_classes = {}
    for f in factory_files:
        path = os.path.join(directory, f)
        relative_import_string = "." + inspect.getmodulename(path)
        module = import_module(relative_import_string, package=__package__)
        for name in dir(module):
            obj = getattr(module, name)
            if inspect.isclass(obj):
                if issubclass(obj, GoalSamplingStrategy):
                    gss_classes[name] = obj

    return gss_classes


def string_to_strategy(name):
    """Get goal sampling strategy class by name."""

    built_in = {"episode", "future", "final"}

    if name in built_in:
        return name
    else:
        return get_goal_sampling_strategy_classes()[name]()
