import argparse
import json
import numpy as np
import torch
import os

from .core import load_params, run_session, get_env_and_graph


def run(dir_path, torch_num_threads=None):
    # load parameters from json files
    run_params, graph_params, varied_hps = load_params(dir_path)
    base_params = {
        "run_params": run_params,
        "graph_params": graph_params
    }

    if os.path.isdir(os.path.join(dir_path, "state")):
        with open(os.path.join(dir_path, "state", "step.json"), "r") as json_file:
            step = json.load(json_file)["step"]
    else:
        step = 0

    print("Current step: ", step)
    
    log_dir = os.path.join(dir_path, "log")
   
    params = base_params
    varied_params = varied_hps

    # seed numpy and pytorch
    np.random.seed(run_params["seed"])
    torch.manual_seed(run_params["seed"])

    env, graph = get_env_and_graph(run_params, graph_params)

    # load graph state in case the run has been executed before
    state_dir = os.path.join(dir_path, "state")
    if os.path.isdir(state_dir):
        print("Loading state of graph.")
        graph.load_state(dir_path=state_dir) 
        # also override logs with saved version in case the process was killed
        # before the state could be saved
        log_state_dir = os.path.join(dir_path, "state", "log")
        if os.path.isdir(log_dir):
            shutil.rmtree(log_dir)
        shutil.copytree(log_state_dir, log_dir)

    # save parameters in json file
    os.makedirs(dir_path, exist_ok = True)

    run_params_path = os.path.join(dir_path, "run_params.json")
    with open(run_params_path, "w") as run_params_file:
        json.dump(run_params, run_params_file, indent = 4)
    graph_params_path = os.path.join(dir_path, "graph_params.json")
    with open(graph_params_path, "w") as graph_params_file:
        json.dump(graph_params, graph_params_file, indent = 4)
    varied_params_path = os.path.join(dir_path, "varied_hp.json")
    with open(varied_params_path, "w") as varied_params_file:
        json.dump(varied_hps, varied_params_file, indent = 4)

    # run session
    sess_props = run_session(dir_path, graph, env, run_params, step)

    # delete old log directory in state directory
    log_state_dir = os.path.join(dir_path, "state", "log")
    if os.path.isdir(log_state_dir):
        shutil.rmtree(log_state_dir)

    if sess_props["timed_out"]:
        # save the state of the graph (replay buffer, parameters...) in 
        # order to be able to continue training
        print("Saving state of graph.")
        state_dir = os.path.join(dir_path, "state")
        os.makedirs(state_dir, exist_ok = True)
        graph.save_state(os.path.join(state_dir))
        # save a copy of the log directory in the state directory
        shutil.copytree(os.path.join(dir_path, "log"), log_state_dir)
    else:
        # delete state if present
        state_path = os.path.join(dir_path, "state")
        if os.path.isdir(state_path):
            shutil.rmtree(state_path)

    os.makedirs(os.path.join(dir_path, "state"), exist_ok = True)
    with open(os.path.join(dir_path, "state", "step.json"), "w") as json_file:
        json.dump({"step": sess_props["total_step"]}, json_file, indent = 4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform run based on parameters read from json files in "
        "provided directory."
    )
    parser.add_argument(
        "dir",
        default=None,
        help="Directory with json files containing parameters for run."
    )
    parser.add_argument(
        "--torch_num_threads",
        default=None,
        type=int,
        help="Overwrites number of threads to use in pytorch."
    )
    args = parser.parse_args()

    run(args.dir, args.torch_num_threads)
