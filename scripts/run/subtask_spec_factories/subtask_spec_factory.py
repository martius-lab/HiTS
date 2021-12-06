from abc import ABC, abstractmethod
import numpy as np

from graph_rl.aux_rewards import DeltaTAchReward, GoalTolReward, ConstantReward


class SubtaskSpecFactory(ABC):

    @classmethod
    def produce(cls, env, graph_params):
        """Get subtask specifications for all levels but the highest one.

        The highest level is derived from the environment (together with map_to_env_goal)."""

        alg_to_method_dict = {
                "HiTS": cls.get_hits_subtask_specs,
                "HAC": cls.get_hac_subtask_specs
                }

        # assemble list of subtask spec params on levels
        subtask_spec_params_list = [l["subtask_spec_params"] for l in graph_params["level_params_list"]]
        assert len(subtask_spec_params_list) == graph_params["n_layers"]

        # construct algorithm specific subtask specs
        assert graph_params["algorithm"] in alg_to_method_dict
        subtask_specs = alg_to_method_dict[graph_params["algorithm"]](
                env, graph_params["n_layers"], subtask_spec_params_list)

        # add auxiliary rewards
        cls.add_auxiliary_rewards(subtask_specs, subtask_spec_params_list)
        
        return subtask_specs

    @classmethod
    @abstractmethod
    def get_hits_subtask_specs(cls, env, n_layers, subtask_spec_params_list):
        """Get subtask specs for HiTS graph from parameters."""
        pass

    @classmethod
    @abstractmethod
    def get_hac_subtask_specs(cls, env, n_layers, subtask_spec_params_list):
        """Get subtask specs for HAC graph from parameters."""
        pass

    @classmethod
    def add_auxiliary_rewards(cls, subtask_specs, subtask_spec_params_list):
        for spec, spec_params in zip(subtask_specs, subtask_spec_params_list):
            # auxiliary reward for encouraging small goal tolerances
            if ("learn_goal_ach_thresholds" in spec_params and 
                    spec_params["learn_goal_ach_thresholds"]):
                f = lambda x: -(spec_params["goal_tol_rew"]
                        *np.array(list(x.values())).mean())
                aux_rew = GoalTolReward(f)
                spec.add_aux_reward(aux_rew)

            # auxiliary reward for encouraging small episode lengths
            if "weight_delta_t_ach_aux" in spec_params:
                aux_rew = DeltaTAchReward(spec_params["weight_delta_t_ach_aux"])
                spec.add_aux_reward(aux_rew)

            # constant auxiliary reward (e.g. for encouraging temporal abstraction 
            # on lower HiTS levels)
            if "constant_reward" in spec_params:
                aux_rew = ConstantReward(spec_params["constant_reward"])
                spec.add_aux_reward(aux_rew)

    @classmethod
    @abstractmethod
    def get_map_to_env_goal(cls, env):
        """Return map from partial observation to environment goal."""
        pass
