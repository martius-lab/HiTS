import numpy as np

from graph_rl.subtasks import (
    BoxInfoHidingTGSubtaskSpec, BoxInfoHidingSPSubtaskSpec, EnvSPSubtaskSpec, 
    EnvRMSubtaskSpec
)

from .subtask_spec_factory import SubtaskSpecFactory

class BoxSubtaskSpecFactory(SubtaskSpecFactory):
    
    @classmethod
    def get_indices_and_factorization(cls, subtask_spec_params, level):
        """Determines observation and goal space and factorization of goal space.
        
        Returns
        partial_obs_indices: Indices of components of "observation" item of
            observation space that are exposed to the level.
        goal_indices: Indices of components of "observation" item of
            observation space that comprise the goal space.
        factorization: List of lists of indices which define the subspaces
            of the goal space in which the Euclidean distance is used.
        """

        pass

    @classmethod
    def get_map_to_env_goal(cls, env):
        """Return map from partial observation to environment goal."""
        pass

    @classmethod
    def get_map_to_subgoal_and_subgoal_space(cls, env):
        """Return map from partial observation to environment goal and subgoal space.
        
        Returns None by default which means the components specified in 
        goal_indices are simply copied from the partial observation into 
        the subgoal."""
        return None

    @classmethod
    def get_hits_subtask_spec(cls, env, subtask_spec_params, level, delta_t_max):
        # get specification of partial obs and goal space for this level
        partial_obs_indices, goal_indices, factorization = cls.get_indices_and_factorization(
            env = env, 
            subtask_spec_params = subtask_spec_params, 
            level = level
        )
        # if a custom map to subgoal is specified, specify custom subtask specs class
        return_value = cls.get_map_to_subgoal_and_subgoal_space(env)
        if return_value is not None:
            map_to_subgoal, subgoal_space = return_value
            class CustomTGSubtaskSpec(BoxInfoHidingTGSubtaskSpec):

                def map_to_goal(self, partial_obs):
                    return map_to_subgoal(partial_obs)

                @property
                def goal_space(self):
                    return subgoal_space

            subtask_spec_class = CustomTGSubtaskSpec
        else:
            subtask_spec_class = BoxInfoHidingTGSubtaskSpec

        # create subtask specs for this level
        subtask_specs = subtask_spec_class(
            delta_t_max=delta_t_max,
            goal_achievement_threshold=subtask_spec_params["goal_achievement_threshold"], 
            partial_obs_indices=partial_obs_indices,
            goal_indices=goal_indices, 
            factorization=factorization,
            delta_t_min=subtask_spec_params.get("delta_t_min",0.),
            env=env
        ) 
        return subtask_specs

    @classmethod
    def get_hits_subtask_specs(cls, env, n_layers, subtask_spec_params_list):
        """Get subtask specs for HiTS graph from parameters."""

        delta_t_max_list = [spec_params["delta_t_max"] for spec_params in subtask_spec_params_list[:-1]]

        # calculate delta_t_max on second highest level if not specified
        if delta_t_max_list[-1] == -1:
            delta_t_max_list[-1] = env.max_episode_length/subtask_spec_params_list[-1]["max_n_actions"]

        # create subtask specs on individual levels
        subtask_specs = []
        zipped_lists = zip(delta_t_max_list, subtask_spec_params_list, range(n_layers))
        # levels zero to n_layers - 1
        for delta_t_max, spec_params, l in zipped_lists:
            subtask_specs.append(cls.get_hits_subtask_spec(env, spec_params, l, delta_t_max))

        # highest level gets subtask spec based on environment
        max_n_actions_highest = subtask_spec_params_list[-1]["max_n_actions"]
        if max_n_actions_highest == -1:
            max_n_actions_highest = env.max_episode_length
        elif max_n_actions_highest == "None":
            max_n_actions_highest = None
        if "constant_failure_return" in subtask_spec_params_list[-1]:
            constant_failure_return_highest = subtask_spec_params_list[-1]["constant_failure_return"]
        else:
            constant_failure_return_highest = False
        if ("type" in subtask_spec_params_list[-1] 
            and subtask_spec_params_list[-1]["type"] == "EnvRMSubtaskSpec"):
            highest_spec = EnvRMSubtaskSpec(env, max_n_actions_highest)
        else:
            highest_spec = EnvSPSubtaskSpec(max_n_actions_highest, env, cls.get_map_to_env_goal(env), 
                    constant_failure_return_highest)
        subtask_specs.append(highest_spec)

        return subtask_specs

    @classmethod
    def get_hac_subtask_spec(cls, env, subtask_spec_params, level, max_n_act):
        # get specification of partial obs and goal space for this level
        partial_obs_indices, goal_indices, factorization = cls.get_indices_and_factorization(
            env = env, 
            subtask_spec_params = subtask_spec_params, 
            level = level
        )
        # if a custom map to subgoal is specified, specify custom subtask specs class
        return_value = cls.get_map_to_subgoal_and_subgoal_space(env)
        if return_value is not None:
            map_to_subgoal, subgoal_space = return_value
            class CustomSPSubtaskSpec(BoxInfoHidingSPSubtaskSpec):

                def map_to_goal(self, partial_obs):
                    return map_to_subgoal(partial_obs)

                @property
                def goal_space(self):
                    return subgoal_space

                @property
                def parent_action_space(self):
                    return subgoal_space

            subtask_spec_class = CustomSPSubtaskSpec
        else:
            subtask_spec_class = BoxInfoHidingSPSubtaskSpec

        # create subtask specs for this level
        subtask_specs = subtask_spec_class(
            max_n_actions = max_n_act, 
            goal_achievement_threshold = subtask_spec_params["goal_achievement_threshold"], 
            partial_obs_indices = partial_obs_indices,
            goal_indices = goal_indices, 
            factorization = factorization,
            env = env
        ) 
        return subtask_specs

    @classmethod
    def get_hac_subtask_specs(cls, env, n_layers, subtask_spec_params_list):
        """Get subtask specs for HAC graph from parameters."""

        max_n_actions_list = [spec_params["max_n_actions"] for spec_params in subtask_spec_params_list]

        # calculate max_n_actions on lowest level if not specified
        if max_n_actions_list[0] == -1:
            max_n_actions_list[0] = env.max_episode_length/np.prod(max_n_actions_list[1:])

        subtask_specs = []
        zipped_lists = zip(max_n_actions_list[:-1], subtask_spec_params_list, range(n_layers))
        # levels zero to n_layers - 1
        for max_n_act, spec_params, l in zipped_lists:
            subtask_specs.append(cls.get_hac_subtask_spec(env, spec_params, l, max_n_act))

        # highest level gets subtask spec based on environment
        max_n_actions_highest = subtask_spec_params_list[-1]["max_n_actions"]
        if max_n_actions_highest == -1:
            max_n_actions_highest = env.max_episode_length
        elif max_n_actions_highest == "None":
            max_n_actions_highest = None
        if "constant_failure_return" in subtask_spec_params_list[-1]:
            constant_failure_return_highest = subtask_spec_params_list[-1]["constant_failure_return"]
        else:
            constant_failure_return_highest = False
        if ("type" in subtask_spec_params_list[-1] 
            and subtask_spec_params_list[-1]["type"] == "EnvRMSubtaskSpec"):
            highest_spec = EnvRMSubtaskSpec(env, max_n_actions_highest)
        else:
            highest_spec = EnvSPSubtaskSpec(max_n_actions_highest, env, cls.get_map_to_env_goal(env), 
                    constant_failure_return_highest)
        subtask_specs.append(highest_spec)
        
        return subtask_specs
