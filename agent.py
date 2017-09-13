import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.state = {}
        # TODO: Initialize any additional variables here
        self.epsilon = 0.5
        self.alpha = 0.6
        self.gamma = 0.2
        self.decayRate = 0.001
        self.actions = [None, 'forward','left','right']
        self.QValues = {}

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = self.build_state(inputs)
        
        # TODO: Select action according to your policy
        #actions = (None, 'forward', 'left', 'right') 
        #action = random.choice(actions)
        action = self.action_policy(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        self.update_qValue(self.state,action,reward)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


    def build_state(self, inputs):
        
        return {
        "light": inputs["light"],
        "oncoming": inputs["oncoming"],
        "left": inputs["left"],
        "right": inputs["right"],
        "location": self.next_waypoint        
        }
        
        
    def action_policy(self, state):
        
        if random.random() < self.epsilon:
            self.epsilon -= self.decayRate
            return random.choice(self.actions)
        
        max_value = 0
        best_action = self.actions[0]
        
        for action in self.actions:
            qVal = self.getQValue(state,action)
            if qVal > max_value:
                max_value = qVal
                best_action = action
            elif max_value == qVal:
                best_action = random.choice([best_action,action])
                
        return best_action
        
        
    def getQValue(self, state, action):
        
        dictKey = self.getKey(state,action)
        #print dictKey
        
        if dictKey in self.QValues:
            return self.QValues[dictKey]
        
        return 0
        
    def max_qValue(self, state):
        
        max_qVal = 0        
        for action in self.actions:
            qVal = self.getQValue(state,action)
            if qVal > max_qVal:
                max_qVal = qVal
                
        return max_qVal
            
        
    def update_qValue(self, state, action, reward):
        
        key = self.getKey(state,action)
        value = self.getQValue(state,action)
        
        inputs = self.env.sense(self)
        self.next_waypoint = self.planner.next_waypoint()
        new_state = self.build_state(inputs)
        
        learned_value = reward + (self.gamma * self.max_qValue(new_state))
        new_qValue = value + (self.alpha *(learned_value - value))
        
        self.QValues[key] = new_qValue
        
        
        
    def getKey(self,state,action):
        
        return "{} --> {} --> {} --> {} --> {} --> {}".format(state["light"],state["oncoming"],state["left"],state["right"],state["location"],action)
        
        
    

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
