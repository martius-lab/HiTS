"""Methods for loading data from CSV files created by GraphRL."""

import os
from itertools import compress

import numpy as np

from graph_rl import Session
from graph_rl.subtasks.shortest_path_subtask import ShortestPathSubtask
from graph_rl.subtasks.timed_goal_subtask import TimedGoalSubtask


# how to read logfiles with a given prefix
prefix_to_read_method = {
    "hac_node": ShortestPathSubtask.read_logfiles,
    "hits_node": TimedGoalSubtask.read_logfiles
}


def load_log_data(experiment_dir):
    """Loads data from CSV files into dict. 

    Returns a dict organized according to the hierarchy:
    logfile->mode(test/train)->quantity"""

    # load csv log data
    log_path = os.path.join(experiment_dir, "log")
    csv_files = [fn for fn in os.listdir(log_path)
            if fn.endswith(".csv")]
    data = {
        "session": Session.read_logfiles(log_path)
        }
    for fn in csv_files:
        for prefix, read in prefix_to_read_method.items():
            if fn.startswith(prefix):
                mode = fn.split("_")[-1].split(".")[0]
                if mode == "train":
                    name = "_".join(fn.split("_")[:-1])
                    fns = {
                            "train": os.path.join(log_path, name + "_train.csv"), 
                            "test": os.path.join(log_path, name + "_test.csv")
                            }
                    data[name] = read(fns)

    return data
