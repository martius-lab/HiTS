from copy import deepcopy

import torch

from graph_rl.models import SACModel, DDPGModel
from graph_rl.utils import listify

from .goal_sampling_strategies.string_to_strategy import string_to_strategy

def get_mlp_models(level_params_list):
    """Get models based on level params and put them into level_algo_kwargs_list."""

    level_algo_kwargs_list = []

    for level_params in level_params_list:
        model_kwargs = level_params["model_kwargs"]
        algo_kwargs = deepcopy(level_params["algo_kwargs"])
        # replace goal sampling strategy name with corresponding class if necessary
        if "goal_sampling_strategy" in algo_kwargs:
            algo_kwargs["goal_sampling_strategy"] = string_to_strategy(algo_kwargs["goal_sampling_strategy"])
        # specify model
        algo_kwargs["model"] = get_mlp_model(
            **model_kwargs,
            flat_algo = algo_kwargs["flat_algo_name"], 
            activation_fn = torch.nn.ReLU(inplace = False), #TODO: Try tanh?
            squash_low = model_kwargs["squash_low"] if model_kwargs["squash_critics"] else None, 
            squash_high = model_kwargs["squash_high"] if model_kwargs["squash_critics"] else None
            )
        level_algo_kwargs_list.append(algo_kwargs)

    return level_algo_kwargs_list

def get_mlp_model(flat_algo, learning_rate, hidden_layers, 
        activation_fn, squash_critics = False, squash_low = None, 
        squash_high = None, actor_clip_threshold = None, 
        critic_clip_threshold = None, device = "cpu", force_negative = False):

    if flat_algo == "SAC":
        ModelClass = SACModel
    elif flat_algo == "DDPG":
        ModelClass = DDPGModel
    else:
        raise ValueError("Unknown flat RL algorithm: {}".format(flat_algo))

    model = ModelClass(
            hidden_layers_actor = hidden_layers,
            hidden_layers_critics = hidden_layers,
            activation_fns_actor = [activation_fn]*len(hidden_layers),
            activation_fns_critics = [activation_fn]*len(hidden_layers),
            learning_rate_actor = learning_rate, 
            learning_rate_critics = learning_rate, 
            squash_critics = squash_critics, 
            low = squash_low, 
            high = squash_high,
            actor_clip_threshold = actor_clip_threshold, 
            critic_clip_threshold = critic_clip_threshold, 
            force_negative = force_negative, 
            device = device
            )

    return model
