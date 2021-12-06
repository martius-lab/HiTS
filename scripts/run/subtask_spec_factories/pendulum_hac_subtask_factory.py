import numpy as np

from graph_rl.spaces import BoxSpace

from .box_subtask_spec_factory import BoxSubtaskSpecFactory


class PendulumHACSubtaskSpecFactory(BoxSubtaskSpecFactory):

    @classmethod
    def bound_angle(cls, angle):
        """Map to [-pi, pi]."""
        bounded_angle = angle % (2*np.pi)

        if np.absolute(bounded_angle) > np.pi:
            bounded_angle = -(np.pi - bounded_angle % np.pi)

        return bounded_angle

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
        # Like HAC paper use angle in subgoal space instead of cosine and sine
        goal_indices = range(2)
        factorization = [[i] for i in range(2)]
        return partial_obs_indices, goal_indices, factorization

    @classmethod
    def get_map_to_env_goal(cls, env):
        """Return map from partial observation to environment goal."""

        # env goal is equal to joint angles
        n_angles = env.observation_space["observation"].shape[0]//2
        def mapping(partial_obs):
            # invert mapping to cosine and sine of angle
            c = partial_obs[0]
            s = partial_obs[1]
            angle = np.sign(s)*np.arccos(c)

            return np.array([cls.bound_angle(angle), np.clip(partial_obs[2], -15, 15)])
        return mapping

    @classmethod
    def get_map_to_subgoal_and_subgoal_space(cls, env):
        """Return map from partial observation to environment goal and subgoal space."""

        def mapping(partial_obs):
            # invert mapping to cosine and sine of angle
            c = partial_obs[0]
            s = partial_obs[1]
            angle = np.sign(s)*np.arccos(c)

            return np.array([cls.bound_angle(angle), np.clip(partial_obs[2], -15, 15)])

        high = np.array([np.pi] + [15.])
        low = -high
        subgoal_space = BoxSpace(low, high, dtype = np.float32)
        return mapping, subgoal_space
