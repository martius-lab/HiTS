from graph_rl.algorithms import GoalSamplingStrategy
from graph_rl.algorithms import HAC

class Tennis2DGSS(GoalSamplingStrategy):

    def __call__(self, episode_transitions, n_hindsight_goals):
        """Hindsight goal is constructed from achieved goal at contact between ball and ground."""

        indices = []
        for i, trans in enumerate(episode_transitions):
            if isinstance(trans, HAC._Transition):
                achieved_goal = trans.subtask_tr.info["achieved_generalized_goal"]
            else:
                achieved_goal = trans.subtask_tr.info["achieved_generalized_goal"]["goal"]
            if achieved_goal[2] == 1.:
                indices.append(i)

        assert len(indices) <= 1
        return indices

