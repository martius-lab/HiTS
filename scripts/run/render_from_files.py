import argparse

from .core import load_params, render, get_env_and_graph


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description = "Render episodes using final policy of run read from provided directory.")
    parser.add_argument("dir", default = None, help = "Directory containing parameters and final policy of run.")
    parser.add_argument("--model_params_path", default = None, help = "Path to model parameters.")
    parser.add_argument("--torch_num_threads", default = None, type = int, help = "Overwrites number of threads to use in pytorch.")
    parser.add_argument("--render_frequency", default = 1, type = int, help = "Frequency of rendered training episodes.")
    parser.add_argument("--test", default = False, action = "store_true", help = "Whether to show testing or training episode.")
    parser.add_argument("--learn", default = False, action = "store_true", help = "Whether to learn.")
    parser.add_argument("--do_not_load_policy", default = False, action = "store_true", help = "Whether to learn.")
    parser.add_argument("--no_render", default = False, action = "store_true", help = "Whether to render episodes.")
    parser.add_argument("--tensorboard_logdir", default = None, 
            help = "Directory for tensorboard log. Is created if necessary.")

    args = parser.parse_args()

    # load parameters from json files
    run_params, graph_params, varied_hps = load_params(args.dir)

    if args.torch_num_threads is not None:
        run_params["torch_num_threads"] = args.torch_num_threads

    # create environment and graph
    env, graph = get_env_and_graph(run_params, graph_params)

    # render
    render(args, graph, env, run_params)
