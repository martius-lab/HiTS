import numpy as np

from graph_rl.aux_rewards import PowerReward

from .dict_subtask_spec_factory import DictSubtaskSpecFactory


class PlatformsSubtaskSpecFactory(DictSubtaskSpecFactory):

    @classmethod
    def _get_dicts(cls, subtask_spec_params, level):
        partial_obs_keys = {"position", "velocity", "ang_vel", "platform0", "platform1"}
        goal_keys = {"position", "velocity"}
        thresholds = {
                "position": subtask_spec_params["goal_achievement_threshold"], 
                "velocity": 0.2 
                }
        return partial_obs_keys, goal_keys, thresholds

    @classmethod
    def add_auxiliary_rewards(cls, subtask_specs, subtask_spec_params_list):
        super().add_auxiliary_rewards(subtask_specs, subtask_spec_params_list)
        for spec, spec_params in zip(subtask_specs, subtask_spec_params_list):
            # auxiliary reward to help minimize work done by agent
            if "power_aux_reward" in spec_params and spec_params["power_aux_reward"]:
                c = spec_params["power_aux_reward_factor"]
                map_to_vel = lambda obs: np.array(obs["velocity"][:1])
                map_to_force = lambda obs, action: np.array(action)
                aux_rew = PowerReward(c, map_to_vel, map_to_force)
                spec.add_aux_reward(aux_rew)

    @classmethod
    def get_map_to_env_goal(cls, env):
        """Return map from partial observation to environment goal."""
        # env goal corresponds to position
        return lambda partial_obs: partial_obs["position"]
