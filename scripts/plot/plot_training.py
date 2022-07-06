import argparse
import copy
import json
import os

import matplotlib.pyplot as plt

from .load_log_data import load_log_data
from .utils import plot_quantity_in_axis


def plot_return(data, mode, plot_path, smoothing_scale=50000):
    """Plot episode return.

    Note that the episode might end prematurely if the higher level 
    has a finite action budget (max_n_actions is not None). In this 
    case the return might be high even though the policy does not solve
    the task (if the reward is negative)."""

    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("steps")
    ax.set_ylabel("return")
    steps = data["session"][mode]["step"]
    returns = data["session"][mode]["return"]
    plot_quantity_in_axis(steps, returns, ax, label=None, smoothing_scale=smoothing_scale) 
    fig.savefig(plot_path, bbox_inches="tight")


def plot_success(data, mode, plot_path, smoothing_scale=50000):
    """Plot success rate."""

    if "success" in data["session"][mode]:
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("steps")
        ax.set_ylabel("success rate")
        ax.set_ylim(0., 1.)
        steps = data["session"][mode]["step"]
        success = data["session"][mode]["success"]
        plot_quantity_in_axis(steps, success, ax, label=None, smoothing_scale=smoothing_scale) 
        fig.savefig(plot_path, bbox_inches="tight")


def plot_n_subgoals(data, mode, plot_path, smoothing_scale=50000):
    """Plot number of subgoals used in episode."""

    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("steps")
    ax.set_ylabel("#subgoals")
    steps = data["hac_node_layer_1_subtask"][mode]["step"]
    if len(steps) > 0:
        n_subgoals = data["hac_node_layer_1_subtask"][mode]["n_actions"]
        plot_quantity_in_axis(steps, n_subgoals, ax, label=None, smoothing_scale=smoothing_scale) 
        fig.savefig(plot_path, bbox_inches="tight")


def plot_training(data, plot_dir, smoothing_scale):
    for mode in ["train", "test"]:
        # return
        return_path = os.path.join(plot_dir, f"return_{mode}.pdf")
        plot_return(data, mode, return_path, smoothing_scale)
        # success rate
        success_path = os.path.join(plot_dir, f"success_{mode}.pdf")
        plot_success(data, mode, success_path, smoothing_scale)
        # number of subgoals
        n_subgoals_path = os.path.join(plot_dir, f"n_subgoals_{mode}.pdf")
        plot_n_subgoals(data, mode, n_subgoals_path, smoothing_scale)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training curves and store them in subdirectory 'plots' of experiment directory.")
    parser.add_argument("experiment_dir", type=str, help="Path to directory with parameter files and 'log' subdirectory.")
    parser.add_argument("--smoothing_scale", type=float, help="Scale over which curves will be smoothed (in steps).", 
        default=10000)
    args = parser.parse_args()

    plot_dir = os.path.join(args.experiment_dir, "plots")
    os.makedirs(plot_dir, exist_ok = True)

    # load data
    data = load_log_data(args.experiment_dir)
    print("Loaded log files: ")
    for k in data:
        print(k)

    # plot
    plot_training(data, plot_dir, args.smoothing_scale)
