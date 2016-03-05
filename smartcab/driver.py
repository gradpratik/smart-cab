import random
import copy
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class State(object):
    def __init__(self, waypoint, light, oncoming, right, left):
        self.data = (waypoint, light, oncoming, right, left)
        self.waypoint = waypoint
        self.light = light
        self.oncoming = oncoming
        self.right = right
        self.left = left

    def __eq__(self, another):
        return hasattr(another, 'data') and self.data == another.data

    def __hash__(self):
        return hash(self.data)

    def __str__(self):
        return str(self.data)

    @staticmethod
    def make(waypoint, inputs):
        return State(waypoint, inputs['light'], inputs['oncoming'], inputs['right'], inputs['left'])

class States(object):
    def __init__(self, actions, initVal):
        values = {action:initVal for action in actions}
        lights = ('green', 'red')
        self.table = {State(waypoint, light, oncoming, right, left) : copy.deepcopy(values)
            for waypoint in actions if waypoint
            for light in lights
            for oncoming in actions
            for right in actions
            for left in actions}

    def __getitem__(self, state):
        return self.table[state]

    def __iter__(self):
        return self.table.__iter__()

    def __len__(self):
        return self.table.__len__()

class Driver(object):
    def __init__(self, actions):
        self.valid_actions = actions
        self.decision = None
        self.alpha = None
        self.gamma = None
        self.q_value = None

    def decide(self, state):
        return None

    def learn(self, time, deadline, state, action, reward, newState):
        pass

class RandomDriver(Driver):
    def __init__(self, actions):
        Driver._init__(self, action)

    def decide(self, state):
        self.decision = 'random'
        return self.random()

    def random(self):
        return random.choice(self.valid_actions)

class GreedyDriver(Driver):
    def __init__(self, actions, initQ=0.0):
        Driver.__init__(self, actions)
        self.q_values = States(actions, initQ)

    def decide(self, state):
        self.decision = 'exploit'
        return self.exploit(state)
        
    def learn(self, time, deadline, state, action, reward, newState):
        self.alpha = 1.0 / (1.0 + time)
        self.gamma = 1.0 / (1.0 + deadline) 
        self.update(state, action, reward, newState)

    def update(self, state, action, reward, newState):
        new_q_value = self.q_values[newState][self.exploit(newState)] if newState.waypoint else 0.0
        self.q_value = (1.0-self.alpha)*self.q_values[state][action]+self.alpha*(reward+self.gamma*new_q_value)
        self.q_values[state][action] = self.q_value

    def exploit(self, state):
        return random.choice([action for action in self.q_values[state] if self.q_values[state][action]==max(self.q_values[state].values())])

class Greedy2Driver(GreedyDriver):
    def __init__(self, actions):
        GreedyDriver.__init__(self, actions, 1.5)

class SmartDriver(GreedyDriver):
    def __init__(self, actions, reporter):
        GreedyDriver.__init__(self, actions)
        self.coverage = reporter.coverage

    def decide(self, state):
        try:
            self.decision = 'explore'
            return self.explore(state)
        except:
            pass
        self.decision = 'exploit'
        return self.exploit(state)

    # expolore will throw if no more to explore
    def explore(self, state):
        return random.choice([action for action in self.coverage[state] if self.coverage[state][action]==0])

    def learn(self, time, deadline, state, action, reward, newState):
        self.alpha = 1.0 / (1.0 + self.coverage[state][action])
        self.gamma = 1.0 / (1.0 + deadline) 
        self.update(state, action, reward, newState)

class Smart2Driver(SmartDriver):
    def __init__(self, actions, reporter):
        SmartDriver.__init__(self, actions, reporter)
        self.reporter = reporter

    def decide(self, state):
        if self.reporter.validation: # last X episodes for validation
            if sum(self.coverage[state].values())==0:
                self.decision = 'skip'
                return None # we have not encounter this state before
            self.decision = 'validation'
            return self.exploit(state)
        return SmartDriver.decide(self, state)

class Smart3Driver(Smart2Driver):
    def decide(self, state):
        if self.reporter.validation: # last X episodes for validation
            if self.exploit(state) is None:
                if random.random() < self.gamma: # use gamma as it increases as deadline decreases
                    self.decision = 'annealing'
                    return self.annealing(state)
        return Smart2Driver.decide(self, state)

    def annealing(self, state):
        try:
            return random.choice([action for action in self.q_values[state] if self.q_values[state][action]>=0 and action is not None])
        except:
            return None

class Reporter:
    def __init__(self, driver_type, actions):
        self.driver_type = driver_type
        self.directory = 'report/{}'.format(driver_type)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.experiment = 0
        self.episode = 0
        self.successCount = 0
        self.coverage = States(actions, 0)
        self.data = pd.DataFrame(columns=[
            'Episode',
            'Time',
            'Deadline',
            'Waypoint',
            'Light',
            'Oncoming',
            'Right',
            'Left',
            'Action',
            'Reward',
            'Coverage',
            'Location',
            'Destination',
            'Decision',
            'Alpha',
            'Gamma',
            'Q value'
        ])
        self.successes = []
        self.successRates = []
        self.coverageRates = []
        self.n_episodes = 1
        self.window = 1
        self.validation = False

    def reset(self, destination):
        self.destination = destination
        self.validation = self.episode >= self.n_episodes - self.window 

    def update(self, time, deadline, state, action, reward, location, driver):
        self.coverage[state][action] += 1
        self.data.loc[len(self.data)] = [
            self.episode,
            time,
            deadline,
            state.waypoint,
            state.light,
            state.oncoming,
            state.right,
            state.left,
            action,
            reward,
            self.coverage[state][action],
            location,
            self.destination,
            driver.decision,
            driver.alpha,
            driver.gamma,
            driver.q_value
        ]
        # report success/failure of this episode
        if self.destination==location:
            self.report(1.0)
        elif deadline==0:
            self.report(0.0)
        
    def report(self, success):
        self.episode += 1
        self.successCount += int(success)
        self.successes.append(success)
        self.successRates.append(100.0*self.successRate())
        self.coverageRates.append(100.0*self.coverageRate())
        print '-' * 40
        print 'Validation' if self.validation else 'Learning'
        print 'Experiment {} Episode {} [{}]'.format(self.experiment, self.episode, 'Success' if success else 'Failure')
        print '#Success/Episodes: {:6.2f}% ({:3d}/{:3d})'.format(100.0*self.successCount/self.episode, self.successCount, self.episode)
        print 'Last {:2d} success  : {:6.2f}%'.format(self.window, self.successRates[-1])
        print 'Coverage         : {:6.2f}%'.format(self.coverageRates[-1])
        print '-' * 40

    def successRate(self):
        n = min(len(self.successes), self.window)
        return sum(self.successes[-n:])/n

    def coverageRate(self):
        return 1.0*sum(1 for state in self.coverage if max(self.coverage[state].values()) > 0)/len(self.coverage)

    def save_to_csv(self):
        self.data.to_csv('{}/{:03d}.csv'.format(self.directory, self.experiment), index=False)

    def drawSuccessRates(self):
        fig = plt.figure()
        plt.plot(range(len(self.successRates)), self.successRates)
        plt.xlabel('# of episodes')
        plt.ylabel('success rate %')
        plt.xlim(0,len(self.successRates)-1)
        plt.ylim(0,101.0)
        fig.savefig('{}/success_{:03d}.png'.format(self.directory, self.experiment), bbox_inches='tight')

    def drawCoverageRates(self):
        fig = plt.figure()
        plt.plot(range(len(self.coverageRates)), self.coverageRates)
        plt.xlabel('# of episodes')
        plt.ylabel('coverage rate %')
        plt.xlim(0,len(self.coverageRates)-1)
        plt.ylim(0,30.0)
        fig.savefig('{}/coverage_{:03d}.png'.format(self.directory, self.experiment), bbox_inches='tight')

class Results(object):
    def __init__(self, driver_type):
        self.driver_type = driver_type
        self.results = pd.DataFrame(columns=['Experiment', 'Success Rate', 'Coverage Rate'])
        self.directory = 'report/{}'.format(driver_type)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def add(self, successRate, coverageRate):
        self.results.loc[len(self.results)] = [len(self.results), successRate, coverageRate]
        
    def show(self):
        successRates = self.results['Success Rate']*100.0
        coverageRates = self.results['Coverage Rate']*100.0
        with open('{}/results.txt'.format(self.directory), 'w+') as f:
            f.write('Results of running ''{}'' driver with {} experiments\n'.format(self.driver_type, len(self.results)))
            f.write('Success Rate  : mean={:6.2f}%\n'.format(successRates.mean()))
            f.write('              : std ={:6.2f}%\n'.format(successRates.std()))
            f.write('              : max ={:6.2f}%\n'.format(successRates.max()))
            f.write('              : min ={:6.2f}%\n'.format(successRates.min()))
            f.write('Coverage Rate : mean={:6.2f}%\n'.format(coverageRates.mean()))
            f.write('              : std ={:6.2f}%\n'.format(coverageRates.std()))
            f.write('              : max ={:6.2f}%\n'.format(coverageRates.max()))
            f.write('              : min ={:6.2f}%\n'.format(coverageRates.min()))
            f.seek(0)
            print f.read()
        self.results.to_csv('{}/results.csv'.format(self.directory), index=False)

