import os
import json
import pprint
import argparse

import gym
import graph_rl
import dyn_rl_benchmarks
import hac_envs

from .graphs import create_graph
from .models import get_mlp_models
from .subtask_spec_factories.string_to_subtask_spec_class import get_subtask_spec_factory_class


def load_params(dir, verbose=False):
    """Load parameters for run from json files.
    
    Does not integrate parameters of individual levels stored 
    in separate files."""

    with open(os.path.join(dir, "run_params.json")) as json_file:
        run_params = json.load(json_file)
    with open(os.path.join(dir, "graph_params.json")) as json_file:
        graph_params = json.load(json_file)
    with open(os.path.join(dir, "varied_hp.json")) as json_file:
        varied_hps = json.load(json_file)

    if verbose:
        pp = pprint.PrettyPrinter(indent = 4)
        print("Varied hyperparameters: ")
        pp.pprint(varied_hps)
        print("Run parameters:")
        pp.pprint(run_params)
        print("Graph parameters:")
        pp.pprint(graph_params)

    return run_params, graph_params, varied_hps


def get_env_and_graph(run_params, graph_params):
    """Get env from gym and construct Graph_RL graph."""

    env_name = run_params["env"]
    env = gym.make(env_name)

    # specifiy subtask specs
    subtask_spec_cl_name = graph_params["subtask_spec_factory"]
    subtask_spec_cl = get_subtask_spec_factory_class(subtask_spec_cl_name)
    subtask_specs = subtask_spec_cl.produce(env, graph_params)

    # get models (actors, critics)
    level_algo_kwargs_list = get_mlp_models(graph_params["level_params_list"])

    # create graph
    graph = create_graph(env, graph_params, run_params, subtask_specs, level_algo_kwargs_list)

    return env, graph


def run_session(dir, graph, env, run_params, total_step_init=0, 
        callback=None):
    """Run session for run and save resulting policy."""

    sess = graph_rl.Session(graph, env)

    # directory for saving the model parameters
    save_directory = os.path.join(dir, "model")
    os.makedirs(save_directory, exist_ok = True)

    if "model_save_frequency" in run_params:
        frequ = run_params["model_save_frequency"]

        class Cb_after_train_episode:
            def __init__(self):
                self.last_model_save = 0

            # callback is executed after each training episode
            def __call__(self, graph, sess_info, ep_return, graph_done):
                if sess_info.total_step - self.last_model_save >= frequ:
                    self.last_model_save = sess_info.total_step
                    # save model params
                    steps_in_k = int(sess_info.total_step/1000)
                    save_path = os.path.join(save_directory, f"params_{steps_in_k}k.pt")
                    graph.save_parameters(save_path)
                # external part of callback
                if callback is not None:
                    callback(graph, sess_info, ep_return, graph_done)

        cb_after_train_episode = Cb_after_train_episode()
    else:
        cb_after_train_episode = None

    tensorboard_log = False if "tensorboard_log" not in run_params else run_params["tensorboard_log"]

    sess_props = sess.run(
            n_steps=run_params["n_steps"], 
            max_runtime=run_params["max_runtime"]*60. if "max_runtime" in run_params else None, 
            learn=True, 
            render=False, 
            test=True, 
            test_render=False, 
            tensorboard_logdir=os.path.join(dir, "tensorboard") if tensorboard_log else None, 
            run_name=None, 
            test_frequency=run_params["test_frequency"], 
            test_episodes=run_params["n_test_episodes"], 
            csv_logdir=os.path.join(dir, "log"), 
            torch_num_threads=run_params.get("torch_num_threads", None),
            append_run_name_to_log_paths=False, 
            cb_after_train_episode=cb_after_train_episode, 
            total_step_init=total_step_init, 
            append_to_logfiles=total_step_init > 0, 
            success_reward=run_params.get("success_reward", None))

    # save model params
    save_path = os.path.join(save_directory, "params.pt")
    graph.save_parameters(save_path)

    return sess_props


def render(args, graph, env, run_params):
    """Render episodes, learning and logging optional."""

    # load model params
    if not args.do_not_load_policy:
        if args.model_params_path is None:
            load_path = os.path.join(args.dir, "model", "params.pt")
            graph.load_parameters(load_path)
        else:
            graph.load_parameters(args.model_params_path)

    # run session
    sess = graph_rl.Session(graph, env)
    sess.run(
            n_steps = run_params["n_steps"], 
            learn = args.learn, 
            render = not args.no_render, 
            test = args.test, 
            test_render = not args.no_render, 
            render_frequency = args.render_frequency, 
            test_render_frequency = args.render_frequency, 
            run_name = None, 
            test_frequency = 1, 
            test_episodes = 1, 
            torch_num_threads = run_params.get("torch_num_threads", None), 
            tensorboard_logdir = args.tensorboard_logdir, 
            append_run_name_to_log_paths = True, 
            train = not args.test,
            success_reward=run_params.get("success_reward", None))
