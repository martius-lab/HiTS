import numpy as np

from graph_rl.subtasks import (DictInfoHidingSPSubtaskSpec, 
        DictInfoHidingTGSubtaskSpec, DictInfoHidingTolTGSubtaskSpec, EnvSPSubtaskSpec)

from .dict_subtask_spec_factory import DictSubtaskSpecFactory

class Tennis2DSubtaskSpecFactory(DictSubtaskSpecFactory):

    @classmethod
    def _get_dicts(cls, subtask_spec_params, level):
        angle_threshold = subtask_spec_params["goal_achievement_threshold"]["angle_threshold"]
        angular_vel_threshold = subtask_spec_params["goal_achievement_threshold"]["angular_vel_threshold"]

        # Let lower level see only the state of the robot, not of the ball
        partial_obs_keys = ({f"joint_{i}_angle" for i in range(3)} |
                {f"joint_{i}_angular_vel" for i in range(3)})
                
        goal_keys = partial_obs_keys
        thresholds = {
            **{f"joint_{i}_angle": angle_threshold for i in range(3)}, 
            **{f"joint_{i}_angular_vel": angular_vel_threshold for i in range(3)}
            }

        return partial_obs_keys, goal_keys, thresholds

    @classmethod
    def get_map_to_env_goal(cls, env):
        """Return map from partial observation to environment goal."""
        # env goal corresponds to position
        return env.map_to_achieved_goal
