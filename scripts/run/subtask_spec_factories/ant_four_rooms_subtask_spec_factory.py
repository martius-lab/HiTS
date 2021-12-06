import numpy as np

from graph_rl.spaces import BoxSpace

from .box_subtask_spec_factory import BoxSubtaskSpecFactory


class AntFourRoomsSubtaskSpecFactory(BoxSubtaskSpecFactory):

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

        n_obs = env.observation_space["observation"].shape[0]
        partial_obs_indices = range(n_obs)
        goal_indices = list(range(3)) + [n_obs//2, n_obs//2 + 1]
        # same as in original HAC paper
        factorization = [[i] for i in range(5)]
        return partial_obs_indices, goal_indices, factorization

    @classmethod
    def get_map_to_env_goal(cls, env):
        """Return map from partial observation to environment goal."""

        # env goal is equal to x, y, z coordinates of torso
        def mapping(partial_obs):
            return partial_obs[:3]
        return mapping

    @classmethod
    def get_map_to_subgoal_and_subgoal_space(cls, env):
        """Return map from partial observation to environment goal and subgoal space."""

        n_coords = env.observation_space["observation"].shape[0]//2
        def mapping(partial_obs):
            pos = np.concatenate((partial_obs[:2], np.clip(partial_obs[2:3], -np.inf, 1.)))
            vels = np.clip(partial_obs[n_coords:n_coords + 2], -3.0, 3.0)
            return np.concatenate((pos, vels))

        high = np.array([9.5]*2 + [1.] + [3.]*2)
        low = -high
        subgoal_space = BoxSpace(low, high, dtype = np.float32)
        return mapping, subgoal_space
