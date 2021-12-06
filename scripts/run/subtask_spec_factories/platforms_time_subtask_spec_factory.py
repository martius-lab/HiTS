from .platforms_subtask_spec_factory import PlatformsSubtaskSpecFactory

class PlatformsTimeSubtaskSpecFactory(PlatformsSubtaskSpecFactory):

    @classmethod
    def _get_dicts(cls, subtask_spec_params, level):
        partial_obs_keys = {"position", "velocity", "ang_vel", "platform0", "platform1", "time"}
        goal_keys = {"position", "velocity", "time"}
        thresholds = {
                "position": subtask_spec_params["goal_achievement_threshold"], 
                "velocity": 0.2, 
                "time": 2.0/500.0,
                }
        return partial_obs_keys, goal_keys, thresholds

    @classmethod
    def get_hits_subtask_specs(cls, env, n_layers, subtask_spec_params_list):
        """Get subtask specs for HiTS graph from parameters."""
        raise NotImplementedError("For version of platforms environment with time in observation HiTS shouldn't be used.")
