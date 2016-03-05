import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import sys
from driver import State, RandomDriver, GreedyDriver, Greedy2Driver, SmartDriver, Smart2Driver, Smart3Driver, Reporter, Results

driver_type = ''

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.debugOn = False
        self.reporter = Reporter(driver_type, self.env.valid_actions)
        if driver_type=='random':
            self.driver = RandomDriver(self.env.valid_actions)
        elif driver_type=='greedy':
            self.driver = GreedyDriver(self.env.valid_actions)
        elif driver_type=='greedy2':
            self.driver = Greedy2Driver(self.env.valid_actions)
        elif driver_type=='smart':
            self.driver = SmartDriver(self.env.valid_actions, self.reporter)
        elif driver_type=='smart2':
            self.driver = Smart2Driver(self.env.valid_actions, self.reporter)
        elif driver_type=='smart3':
            self.driver = Smart3Driver(self.env.valid_actions, self.reporter)
        else:
            self.driver = Smart2Driver(self.env.valid_actions, self.reporter)

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.reporter.reset(destination)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = State.make(self.next_waypoint, inputs)
        
        # TODO: Select action according to your policy
        action = self.driver.decide(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        newWaypoint = self.planner.next_waypoint()
        newState = State.make(newWaypoint, self.env.sense(self))
        self.driver.learn(t, deadline, self.state, action, reward, newState)

        location = self.env.agent_states[self]['location']
        if self.debugOn:
            print "LearningAgent.update(): deadline={} inputs={} action={} reward={} location={}".format(deadline, inputs, action, reward, location)  # [debug]
        self.reporter.update(t, deadline, self.state, action, reward, location, self.driver)

def run(experiment, n_episodes, window, results):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    a.reporter.experiment = experiment
    a.reporter.n_episodes = n_episodes
    a.reporter.window = window
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=n_episodes)  # press Esc or close pygame window to quit

    # reporter
    a.reporter.save_to_csv()
    a.reporter.drawSuccessRates()
    a.reporter.drawCoverageRates()
    results.add(a.reporter.successRate(), a.reporter.coverageRate())

if __name__ == '__main__':
    # conducts given number of experiments (each of which has the specified number of episodes)
    driver_type = 'smart2'
    n_experiments = 1
    n_episodes = 10
    window = 10 # number of the last episodes used for validation
    if len(sys.argv)>1:
        driver_type = sys.argv[1]
    if len(sys.argv)>2:
        n_experiments = int(sys.argv[2])
    if len(sys.argv)>3:
        n_episodes = int(sys.argv[3])
    if len(sys.argv)>4:
        window = int(sys.argv[4])
    window = min(window, n_episodes)
    results = Results(driver_type)
    for experiment in range(n_experiments):
        run(experiment, n_episodes, window, results)
    results.show()

