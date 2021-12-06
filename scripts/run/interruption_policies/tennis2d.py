def tennis_2d_ip(env_obs, subtask_obs):
    """Return True if the ball bounces off the floor for the 2nd time."""

    return env_obs["achieved_goal"][2] == 1.
