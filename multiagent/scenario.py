import numpy as np


class BaseScenario(object):
    """defines scenario upon which the world is built"""

    MIN_COVER_DIST = 0.1

    @property
    def has_shaped_reward(self):
        return False

    def make_world(self):
        """create elements of the world"""
        raise NotImplementedError()

    def reset_world(self, world):
        """create initial conditions of the world"""
        raise NotImplementedError()

    def done(self, agent, world):
        return False

    @staticmethod
    def dist(agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        return dist

    def is_collision(self, agent1, agent2):
        if agent1 is agent2:
            return False
        dist = self.dist(agent1, agent2)
        dist_min = agent1.size + agent2.size
        return dist < dist_min

    def does_cover(self, agent1, agent2):
        dist = self.dist(agent1, agent2)
        return dist < self.MIN_COVER_DIST


class ShapedRewardScenario(BaseScenario):

    @property
    def has_shaped_reward(self):
        return True

    def reward(self, agent, world):
        return self.shaped_reward(agent, world)

    def _reward(self, agent, world, shaped):
        raise NotImplementedError()

    def shaped_reward(self, agent, world):
        return self._reward(agent, world, shaped=True)

    def sparse_reward(self, agent, world):
        return self._reward(agent, world, shaped=False)
