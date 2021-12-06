from .design_agent_and_env import design_agent_and_env as design_agent_and_env_ur5
from .example_designs.PENDULUM_LAY_2_design_agent_and_env import design_agent_and_env as design_agent_and_env_pendulum
from .ant_environments.ant_four_rooms_3_levels.design_agent_and_env import design_agent_and_env as design_agent_and_env_ant_four_rooms
from .gym_wrapper import GymWrapper


class Flags():
    def __init__(self):
        self.retrain = True
        self.test = True
        self.show = False
        self.train_only = True
        self.verbose = True
        self.layers = None
        self.time_scale = None

def ur5_reacher():
    flags = Flags()
    hac_env = design_agent_and_env_ur5(flags)
    return GymWrapper(hac_env)

def pendulum():
    flags = Flags()
    hac_env = design_agent_and_env_pendulum(flags)
    return GymWrapper(hac_env)

def ant_four_rooms():
    flags = Flags()
    hac_env = design_agent_and_env_ant_four_rooms(flags)
    return GymWrapper(hac_env)


