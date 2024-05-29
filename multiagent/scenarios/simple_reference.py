import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import ShapedRewardScenario


class Scenario(ShapedRewardScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 10
        # add agents
        world.agents = [Agent() for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent %d" % i
            agent.collide = False
        # add landmarks
        world.landmarks = [Landmark() for i in range(3)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        # want other agent to go to the goal landmark
        world.agents[0].goal_a = world.agents[1]
        world.agents[0].goal_b = np.random.choice(world.landmarks)
        world.agents[1].goal_a = world.agents[0]
        world.agents[1].goal_b = np.random.choice(world.landmarks)
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])
        # random properties for landmarks
        world.landmarks[0].color = np.array([0.75, 0.25, 0.25])
        world.landmarks[1].color = np.array([0.25, 0.75, 0.25])
        world.landmarks[2].color = np.array([0.25, 0.25, 0.75])
        # special colors for goals
        world.agents[0].goal_a.color = world.agents[0].goal_b.color
        world.agents[1].goal_a.color = world.agents[1].goal_b.color
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def _reward(self, agent, world, shaped):
        reward = 0.0
        for agent in world.agents:
            if agent.goal_a is None or agent.goal_b is None:
                continue
            if shaped:
                dist = self.dist(agent.goal_a, agent.goal_b)
                reward -= dist
            else:
                reward += 1.0 if self.does_cover(agent.goal_a, agent.goal_b) else 0.0
        if shaped and self.done(agent, world):
            reward += 10.0
        return reward

    def done(self, agent, world):
        is_done = lambda agent: self.does_cover(agent.goal_a, agent.goal_b)
        return all(is_done(a) for a in world.agents)

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
        return np.concatenate(
            [agent.state.p_vel] + entity_pos + [agent.goal_b.color] + comm
        )
