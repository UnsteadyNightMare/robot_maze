import random
import numpy as np
from collections import defaultdict

class Robot(object):

    def __init__(self, maze, alpha=0.5, gamma=0.9, epsilon0=0.5):

        self.maze = maze
        self.valid_actions = self.maze.valid_actions
        self.state = None
        self.action = None

        # Set Parameters of the Learning Robot
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon0 = epsilon0
        self.epsilon = epsilon0
        self.t = 0

        self.Qtable = {}
        self.reset()

    def reset(self):
        """
        Reset the robot
        """
        self.state = self.sense_state()
        self.create_Qtable_line(self.state)

    def set_status(self, learning=False, testing=False):
        """
        Determine whether the robot is learning its q table, or
        exceuting the testing procedure.
        """
        self.learning = learning
        self.testing = testing

    def update_parameter(self):
        """
        Some of the paramters of the q learning robot can be altered,
        update these parameters when necessary.
        """
        if self.testing:
            # TODO 1. No random choice when testing
            epsilon = 0.01
        else:
            # TODO 2. Update parameters when learning
            self.t += 1
#             print(self.t)
#             if self.t < 500:
#                 self.epsilon = 0.5
#             else:
            self.epsilon = self.epsilon0 / (self.t / 200 + 1)
#             self.epsilon = 0.5

#         return self.epsilon

    def sense_state(self):
        """
        Get the current state of the robot. In this
        """

        # TODO 3. Return robot's current state
        return self.maze.sense_robot()

    def create_Qtable_line(self, state):
        """
        Create the qtable with the current state
        """
        # TODO 4. Create qtable with current state
        # Our qtable should be a two level dict,
        # Qtable[state] ={'u':xx, 'd':xx, ...}
        # If Qtable[state] already exits, then do
        # not change it.
        # if not self.Qtable[state]:
#         if not all(self.Qtable[state].values()):
        
#         if np.sum(self.Qtable[state].values()):
#         print(self.Qtable)
        if state not in self.Qtable.keys():
            self.Qtable[state] = {}
            map_action = {'u':'d', 'd':'u', 'l':'r', 'r':'l'}
            for a in self.valid_actions:
    #                 print(self.sense_state())
    #                 print(a)
                reward = self.maze.move_robot(a)
                if reward != -10:
                    self.maze.move_robot(map_action[a])
    #                 self.maze.place_robot(robot_loc=state)
                next_state = self.sense_state()
    #                 print(next_state)
    #             print(self.Qtable[state])
    #             print(np.max(self.Qtable[next_state].values()))
    #                 print(state)
    #                 print(next_state)
                self.Qtable[state][a] = reward
    #                 print(self.Qtable[state])
                next_state = state

    def choose_action(self):
        """
        Return an action according to given rules
        """
        def is_random_exploration():

            # TODO 5. Return whether do random choice
            # hint: generate a random number, and compare
            # it with epsilon
            random_num = random.random()
            return random_num < self.epsilon

        if self.learning:
            if is_random_exploration():
                # TODO 6. Return random choose aciton
                probs = np.ones(len(self.valid_actions)) / len(self.valid_actions)
                action = np.random.choice(self.valid_actions, p=probs)
#                 print('11111')
                return action
            else:
                # TODO 7. Return action with highest q value
                probs = np.zeros(len(self.valid_actions)) # 否则选择具有最大 Q 值的动作
                key = sorted(self.Qtable[self.state], key=lambda x:self.Qtable[self.state][x])[-1]
#                 print(self.Qtable[self.state])
                for i, v in enumerate(self.valid_actions):
                    if v == key:
                        probs[i] = 1 
#                 print(probs)
#                 print(random_num)
#                 print('22222')
#                 print(probs)
                action = np.random.choice(self.valid_actions, p=probs)
                return action
        elif self.testing:
            # TODO 7. choose action with highest q value
            probs = np.zeros(len(self.valid_actions)) # 否则选择具有最大 Q 值的动作
            key = sorted(qline, key=lambda x:qline[x])[-1]
            for i, v in enumerate(self.valid_actions):
                if v == key:
                    probs[i] = 1 
#             print(probs)
#             print(random_num)
            action = np.random.choice(actions, p=probs)
            return action
        else:
            # TODO 6. Return random choose aciton
            
            probs = np.ones(len(self.valid_actions)) / len(self.valid_actions)
            action = np.random.choice(self.valid_actions, p=probs)

            return action

    def update_Qtable(self, r, action, next_state):
        """
        Update the qtable according to the given rule.
        """
        if self.learning:
#             Q_next = np.max([i for i in self.Qtable[next_state].values()])
#             while True:
                
#             print('Q_next')
#             print(Q_next)
#             print('action')
#             print(action)
#                 state = self.state
#             Q_state = self.Qtable[self.state][action]
    #                 action_next = self.choose_action()
    #                 reward = self.maze.move_robot(action_next)
    #                 next_state = self.state
#             Q_next = max(self.Qtable[next_state].values())
#             print(Q_next)
#             Q_state = Q_state + self.alpha * (r + self.gamma * Q_next - Q_state)
#             if self.Qtable[self.state][action] > 0:
#                 print(self.Qtable[self.state][action])
#                 print(self.state)
#                 print(self.Qtable[self.state])
#                 state = next_state
#                 if reward == 50:
#                     break
            # TODO 8. When learning, update the q table according
            # to the given rules
            self.Qtable[self.state][action] = self.Qtable[self.state][action] + \
            self.alpha * (r + self.gamma * self.Qtable[next_state][action] - self.Qtable[self.state][action])
#             print(self.Qtable)
    def update(self):
        """
        Describle the procedure what to do when update the robot.
        Called every time in every epoch in training or testing.
        Return current action and reward.
        """
        self.state = self.sense_state() # Get the current state
#         print(self.state)
        self.create_Qtable_line(self.state) # For the state, create q table line

        action = self.choose_action()
#         print(action)# choose action for this state
        reward = self.maze.move_robot(action) # move robot for given action

        next_state = self.sense_state() # get next state
        self.create_Qtable_line(next_state) # create q table line for next state
#         print(next_state)

        if self.learning and not self.testing:
            self.update_Qtable(reward, action, next_state)
#             print(self.update_Qtable)# update q table
            self.update_parameter() # update parameters

        return action, reward
