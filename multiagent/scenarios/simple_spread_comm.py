import itertools as it

import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenarios.simple_spread import Scenario as NoCommScenario


class Scenario(NoCommScenario):
    def make_world(self):
        world = super().make_world()
        for agent in world.agents:
            agent.silent = False
        return world

    def observation(self, agent, world):
        # get positions of all landmarks in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_colors = []
        for entity in world.landmarks:  # world.entities:
            entity_colors.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            if not other.silent:
                comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos + entity_colors + comm)
