from gym.envs.registration import register

register(
        id = "UR5Reacher-v1",
        entry_point = "hac_envs.gym_envs:ur5_reacher"
        )
register(
        id = "PendulumHAC-v1",
        entry_point = "hac_envs.gym_envs:pendulum"
        )
register(
        id = "AntReacher-v1",
        entry_point = "hac_envs.gym_envs:ant_reacher"
        )
register(
        id = "AntFourRooms-v1",
        entry_point = "hac_envs.gym_envs:ant_four_rooms"
        )
