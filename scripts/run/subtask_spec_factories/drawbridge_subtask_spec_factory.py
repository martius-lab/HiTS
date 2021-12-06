from .dict_subtask_spec_factory import DictSubtaskSpecFactory


class DrawbridgeSubtaskSpecFactory(DictSubtaskSpecFactory):

    @classmethod
    def _get_dicts(cls, subtask_spec_params, level):
        partial_obs_keys = {"ship_pos", "ship_vel", "sails_unfurled", "bridge_phase"}
        goal_keys = {"ship_pos", "ship_vel", "sails_unfurled"}
        thresholds = subtask_spec_params["goal_achievement_threshold"]
        return partial_obs_keys, goal_keys, thresholds

    @classmethod
    def get_map_to_env_goal(cls, env):
        """Return map from partial observation to environment goal."""
        # env goal corresponds to position
        return lambda partial_obs: partial_obs["ship_pos"]
