import numpy as np

from graph_rl.spaces import BoxSpace

from .box_subtask_spec_factory import BoxSubtaskSpecFactory


class UR5ReacherSubtaskSpecFactory(BoxSubtaskSpecFactory):

    @classmethod
    def bound_angle(cls, angle):
        bounded_angle = np.absolute(angle) % (2*np.pi)
        if angle < 0:
            bounded_angle = -bounded_angle

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
        goal_indices = range(n_obs)
        # same as in original HAC paper
        factorization = [[i] for i in range(n_obs)]
        return partial_obs_indices, goal_indices, factorization

    @classmethod
    def get_map_to_env_goal(cls, env):
        """Return map from partial observation to environment goal."""

        # env goal is equal to joint angles
        n_angles = env.observation_space["observation"].shape[0]//2
        def mapping(partial_obs):
            return np.array([cls.bound_angle(a) for a in partial_obs[:n_angles]])
        return mapping

    @classmethod
    def get_map_to_subgoal_and_subgoal_space(cls, env):
        """Return map from partial observation to environment goal and subgoal space."""

        n_angles = env.observation_space["observation"].shape[0]//2
        def mapping(partial_obs):
            angles = np.array([cls.bound_angle(a) for a in partial_obs[:n_angles]])
            ang_vels = np.clip(partial_obs[n_angles:], -4.0, 4.0)
            return np.concatenate((angles, ang_vels))

        high = np.array([2.*np.pi]*3 + [4.]*3)
        low = -high
        subgoal_space = BoxSpace(low, high, dtype = np.float32)
        return mapping, subgoal_space
