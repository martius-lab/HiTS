import numpy as np

from graph_rl.spaces import BoxSpace

from .box_subtask_spec_factory import BoxSubtaskSpecFactory


class BallInCupSubtaskSpecFactory(BoxSubtaskSpecFactory):

    @classmethod
    def get_indices_and_factorization(cls, env, subtask_spec_params, level):
        """Determines observation and goal space and factorization of goal space.
        
        Returns
        partial_obs_indices: Indices of components of "observation" item of
            observation space that are exposed to the level.
        goal_indices: Indices of components of "observation" item of
            observation space that comprise the goal space.
        factorization: List of lists of indices which define the subspaces
            of the goal space in which the Euclidean distance is used.
        """

        partial_obs_indices = [0, 1, 4, 5]
        goal_indices = [0, 1, 2, 3]
        factorization = [[i] for i in goal_indices]
        return partial_obs_indices, goal_indices, factorization

    @classmethod
    def get_map_to_env_goal(cls, env):
        """Return map from partial observation to environment goal."""

        def mapping(partial_obs):
            return None
        return mapping

    @classmethod
    def get_map_to_subgoal_and_subgoal_space(cls, env):
        """Return map from partial observation to subgoal and subgoal space."""

        low = np.array([-0.25, -0.28] + [-3.]*2)
        high = np.array([0.25, 0.189] + [3.]*2)
        subgoal_space = BoxSpace(low, high, dtype=np.float32)

        def mapping(partial_obs):
            indices = [0, 1, 2, 3]
            sg = partial_obs[indices]
            sg = np.clip(sg, low, high)
            return sg

        return mapping, subgoal_space
