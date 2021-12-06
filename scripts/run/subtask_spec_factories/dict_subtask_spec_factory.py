import numpy as np

from graph_rl.subtasks import (DictInfoHidingSPSubtaskSpec, 
        DictInfoHidingTGSubtaskSpec, DictInfoHidingTolTGSubtaskSpec, EnvSPSubtaskSpec)

from .subtask_spec_factory import SubtaskSpecFactory


class DictSubtaskSpecFactory(SubtaskSpecFactory):
    """Generates subtask specs from params when partial observation and goal spaces are dict spaces."""

    @classmethod
    def get_hits_subtask_specs(cls, env, n_layers, subtask_spec_params_list):
        """Get subtask specs for HiTS graph from parameters."""
        delta_t_max_list = [spec_params["delta_t_max"] for spec_params in subtask_spec_params_list[:-1]]

        # calculate delta_t_max on second highest level if not specified
        if delta_t_max_list[-1] == -1:
            delta_t_max_list[-1] = env.max_episode_length/subtask_spec_params_list[-1]["max_n_actions"]

        # create subtask specs on individual levels
        subtask_specs = []
        for i, spec_params, delta_t_max in zip(range(n_layers), subtask_spec_params_list[:-1], 
                delta_t_max_list):
            partial_obs_keys, goal_keys, thresholds = cls._get_dicts(spec_params, i)
            norms = cls._get_norms(spec_params, goal_keys, i)
            delta_t_min = spec_params["delta_t_min"] if "delta_t_min" in spec_params else 0.
            if not subtask_spec_params_list[i + 1]["learn_goal_ach_thresholds"]:
                subtask_specs.append(
                        DictInfoHidingTGSubtaskSpec(thresholds, partial_obs_keys, goal_keys, 
                            env, delta_t_max=delta_t_max, delta_t_min=delta_t_min, norms=norms))
            else:
                subtask_specs.append(
                        DictInfoHidingTolTGSubtaskSpec(partial_obs_keys, goal_keys, 
                            env, delta_t_max=delta_t_max, delta_t_min=delta_t_min))

        # highest level gets shortest path subtask spec based on env goal 
        # because it uses HAC
        max_n_actions_highest = subtask_spec_params_list[-1]["max_n_actions"]
        if max_n_actions_highest == -1:
            max_n_actions_highest = env.max_episode_length
        if max_n_actions_highest == "None":
            max_n_actions_highest = None
        highest_spec = EnvSPSubtaskSpec(max_n_actions_highest, env, cls.get_map_to_env_goal(env), 
                subtask_spec_params_list[-1]["constant_failure_return"])
        subtask_specs.append(highest_spec)

        return subtask_specs

    @classmethod
    def _get_norms(cls, subtask_spec_params, goal_keys, level):
        # Use l2 norm by default
        return {key: None for key in goal_keys}

    @classmethod
    def get_hac_subtask_spec_class(cls):
        return DictInfoHidingSPSubtaskSpec

    @classmethod
    def get_hac_subtask_specs(cls, env, n_layers, subtask_spec_params_list):
        """Get subtask specs for HAC graph from parameters."""
        max_n_actions_list = [spec_params["max_n_actions"] for spec_params in subtask_spec_params_list]

        # calculate max_n_actions on lowest level if not specified
        if max_n_actions_list[0] == -1:
            max_n_actions_list[0] = env.max_episode_length/np.prod(max_n_actions_list[1:])

        # create subtask specs on individual levels
        subtask_specs = []
        for max_n_act, spec_params, l in zip(max_n_actions_list[:-1], subtask_spec_params_list, range(len(max_n_actions_list))):
            partial_obs_keys, goal_keys, thresholds = cls._get_dicts(spec_params, l)
            subtask_spec_class = cls.get_hac_subtask_spec_class()
            subtask_specs.append(
                    subtask_spec_class(max_n_act, thresholds, 
                        partial_obs_keys, goal_keys, env))

        # highest level gets shortest path subtask spec based on env goal 
        max_n_actions_highest = subtask_spec_params_list[-1]["max_n_actions"]
        if max_n_actions_highest == -1:
            max_n_actions_highest = env.max_episode_length
        if "constant_failure_return" in subtask_spec_params_list[-1]:
            constant_failure_return_highest = subtask_spec_params_list[-1]["constant_failure_return"]
        else:
            constant_failure_return_highest = False
        highest_spec = EnvSPSubtaskSpec(max_n_actions_highest, env, cls.get_map_to_env_goal(env), 
                constant_failure_return_highest)
        subtask_specs.append(highest_spec)

        return subtask_specs
