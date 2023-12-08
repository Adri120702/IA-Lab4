# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.captureAgents import CaptureAgent
from contest.game import Directions, Actions
from contest.util import nearestPoint

NUM_TRAINING = 0
TRAINING = False

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    NUM_TRAINING = num_training 
    return [eval(first)(first_index), eval(second)(second_index)]
    
    
#########################
######## QLearning ######
#########################


class OffensiveQLearning(CaptureAgent):
    
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.q_values = util.Counter()
        self.alpha = 0.25
        self.discount = 0.75
        self.epsilon = 0.0
        

        '''self.weights = {'closest-food': -2, 
						'bias': 1, 
						'#-of-ghosts-1-step-away': -0.19, 
						'successorScore': -0.03, 
						'eats-food': 9.97}'''

    def register_initial_state(self, game_state):
        """
        Initialize Q-learning parameters.
        """
        self.num_training = NUM_TRAINING
        self.episodes_so_far = 0
        self.weights = self.get_weights(game_state, None)
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def get_q_value(self, game_state, action):
        features = self.getFeatures(game_state, action)
        return features * self.weights

    def compute_value_from_q_values(self, game_state):
        legal_actions = game_state.get_legal_actions(self.index)
        if legal_actions:
            q_values = [self.get_q_value(game_state, action) for action in legal_actions]
            max_q_value = max(q_values)
            return max_q_value
        else:
            return 0.0

    def compute_action_from_q_values(self, game_state):
        legal_actions = game_state.get_legal_actions(self.index)
        best_value = -float("inf")
        legal_actions.remove(Directions.STOP)
        best_action = None
        if legal_actions:
            for action in legal_actions:
                #self.updateWeights(state, action)
                q_value = self.get_q_value(game_state, action)
                if q_value > best_value:
                    best_value = q_value
                    best_action = action
        return best_action

    def choose_action(self, game_state):
        # Pick Action
        legalActions = game_state.get_legal_actions(self.index)
        
        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in legalActions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action
        
        if TRAINING:
            for action in legalActions:
                self.updateWeights(game_state, action)
        
        if len(legalActions) == 0:
            return None
        else:
            if not util.flipCoin(self.epsilon):
            # exploit
                action = self.get_policy(game_state)
            else:
            # explore
                action = random.choice(legalActions)
            return action 

    def get_policy(self, game_state):
        return self.compute_action_from_q_values(game_state)

    def get_value(self, game_state):
        return self.compute_value_from_q_values(game_state)
        
          
    def evaluate(self, game_state, action, nextState, reward):
        disc = self.discount
        learning_rate = self.alpha
        features = self.getFeatures(game_state, action) #Still needs to be done
        qvalue = self.get_q_value(game_state, action)
        next_qvalue = self.get_value(nextState)
        difference = (reward + disc * next_qvalue) - qvalue
        
        for feature in features:
            newWeight = learning_rate * difference * features[feature]
            self.weights[feature] += newWeight
            
    def updateWeights(self, game_state, action):
        nextState = self.get_successor(game_state, action)
        reward = self.getReward(game_state, nextState)  #Still needs to be done
        self.evaluate(game_state, action, nextState, reward)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor
        
    def closest_food(self, pos, food, walls):  #We use breadthFirstSearch to find the distance to the closest food
        matrix = util.Queue()
        matrix.push(([pos[0], pos[1]], 0))
        neigh = []
        while matrix:

            pos, dist = matrix.pop()
            
            if pos in neigh:
                continue
            
            neigh.append(pos)

            if pos in food:
                return dist

            nbrs = Actions.get_legal_neighbors(pos, walls)
                
            for n in nbrs:
                matrix.push((n, dist+1))
        return None
    
    def closest_ghost(self, pos, ghost, walls):  #We use breadthFirstSearch to find the distance to the closest ghost
        matrix = util.Queue()
        matrix.push(([pos[0], pos[1]], 0))
        neigh = []
        while matrix:

            pos, dist = matrix.pop()
            
            if pos in neigh:
                continue
            
            neigh.append(pos)

            if pos in ghost:
                return dist

            nbrs = Actions.get_legal_neighbors(pos, walls)
                
            for n in nbrs:
                matrix.push((n, dist+1))
        return None
    
    def closest_pacman(self, pos, pacman, walls):  #We use breadthFirstSearch to find the distance to the closest pacman
        matrix = util.Queue()
        matrix.push(([pos[0], pos[1]], 0))
        neigh = []
        while matrix:

            pos, dist = matrix.pop()
            
            if pos in neigh:
                continue
            
            neigh.append(pos)

            if pos in pacman:
                return dist

            nbrs = Actions.get_legal_neighbors(pos, walls)
                
            for n in nbrs:
                matrix.push((n, dist+1))
        return None
    
    def distance_capsule(self, pos, capsule, walls):  #We use breadthFirstSearch to find the distance to the capsule
        matrix = util.Queue()
        matrix.push(([pos[0], pos[1]], 0))
        neigh = []
        while matrix:

            pos, dist = matrix.pop()
            
            if pos in neigh:
                continue
            
            neigh.append(pos)

            if pos in capsule:
                return dist

            nbrs = Actions.get_legal_neighbors(pos, walls)
                
            for n in nbrs:
                matrix.push((n, dist+1))
        return None
    
class OffensiveReflexAgent(OffensiveQLearning):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    
    def getFeatures(self, game_state, action):
        # extract the grid of food and wall locations and get the ghost locations
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(game_state).as_list()
        enemies =[successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a.get_position() for a in enemies if not a.is_pacman and a.get_position() != None]
        walls = game_state.get_walls()
        original_pos = game_state.get_agent_state(self.index).get_position()
        my_pos = successor.get_agent_state(self.index).get_position()

        features = util.Counter()

        features["bias"] = 1.0 
        features["#-of-ghosts-1-step-away"] = sum((my_pos) in Actions.get_legal_neighbors(g, walls) for g in ghosts)
        
        # If the pacman is being persecuted it goes to take the capsule
        capsule = self.get_capsules(game_state)
        if len(ghosts) > 0 and capsule: 
            if self.closest_ghost(my_pos, ghosts , walls) < 3:
                state_distance_to_capsule = self.distance_capsule(original_pos, capsule, walls)
                print(state_distance_to_capsule)
                next_state_distance_to_capsule = self.distance_capsule(my_pos, capsule, walls)
                if next_state_distance_to_capsule == None:
                    next_state_distance_to_capsule = 0
                distance_ghost_to_capsule = min([self.distance_capsule(g, capsule, walls) for g in ghosts])
                if next_state_distance_to_capsule < state_distance_to_capsule and state_distance_to_capsule < distance_ghost_to_capsule:
                    features['go-to-capsule'] = 1.0

        # Compute distance to the nearest food
        if len(food_list) > 0: 
            distToFood = self.closest_food(my_pos, food_list , walls)
            features['distance_to_food'] = float(distToFood) / (walls.width * walls.height)
            
        # Compute distance to the nearest ghost
        if len(ghosts) > 0:
            distToGhost = self.closest_ghost(my_pos, ghosts , walls)
            features['distance_to_ghost'] = float(distToGhost) / (walls.width * walls.height)
            
        #Check if pacman eats food
        if not features["#-of-ghosts-1-step-away"] and my_pos in food_list:
            features["eats-food"] = 1.0
            
        # If the pacman is carrying some food and a ghost come close, but it has safe way to its side then it return to save the points of the food
        num_carry = game_state.get_agent_state(self.index).num_carrying
        if num_carry == 0: 
            self.RETURNING = False
        if len(ghosts) > 0:
            if self.get_maze_distance(my_pos, self.start) <  self.get_maze_distance(original_pos, self.start) and  self.get_maze_distance(my_pos, self.start) < min([self.get_maze_distance(g, self.start) for g in ghosts]) and self.closest_ghost(my_pos, ghosts , walls) < 5:
                self.RETURNING = True
            else:
                self.RETURNING = False
        
        if self.RETURNING == True:
            if self.get_maze_distance(my_pos, self.start) <  self.get_maze_distance(original_pos, self.start):
                if num_carry  < 3:
                    features["return-with-food"] = 0
                else:
                    features["return-with-food"] = 1.0
                 
            
        return features

    def get_weights(self, game_state, action):
        return {'distance_to_food': -3.0,
                'distance_to_ghost': 2.0,
                'go-to-capsule': 30.0,
                "return-with-food": 50.0,
                'bias': -9.2,
                '#-of-ghosts-1-step-away': -100.0,
                'eats-food': 11.0}
    
    def getReward(self, game_state, nextState):
        
        reward = 0
        agentPosition = game_state.get_agent_position(self.index)
        NextAgentPosition = nextState.get_agent_position(self.index)
        walls = game_state.get_walls()

        # check if I have updated the score
        if self.get_score(nextState) > self.get_score(game_state):
            diff = self.get_score(nextState) - self.get_score(game_state)
            reward = diff * 10

        # check if food eaten in nextState
        food_list = self.get_food(nextState).as_list()
        distToFood = self.closest_food(agentPosition, food_list, walls)
        NewDistToFood = self.closest_food(NextAgentPosition, food_list, walls)
        #distToFood = min([self.get_maze_distance(agentPosition, food) for food in myFoods])
        #NewDistToFood = min([self.get_maze_distance(NextAgentPosition, food) for food in myFoods])
        
        #if not self.red:
        if NextAgentPosition in food_list: #Will I ate food
            reward = reward + 8
        
        if NewDistToFood < distToFood: #I'm closer to the food
            reward = reward + 4


        # check if I am eaten
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state) if not game_state.get_agent_state(i).is_pacman]
        ghosts = [ghost for ghost in enemies if ghost.get_position() is not None]
        if ghosts:
            minDistGhost = min(self.get_maze_distance(agentPosition, ghost.get_position()) for ghost in ghosts) #Get distance between agent and closest_ghost
            if minDistGhost == 1 and nextState.get_agent_state(self.index).get_position() == self.start: # Checks if I die doeing this move
                reward = -100

        return reward

class DefensiveReflexAgent(OffensiveQLearning):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like. It is not the best or only way to make
    such an agent.
    """

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.my_foods = self.get_food(game_state).as_list()
        self.op_foods = self.get_food_you_are_defending(game_state).as_list()


    def getFeatures(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        # Detecting invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        # Strategic areas to defend
        if my_pos in self.my_foods:
            features['protect_food'] = 1
        if my_pos in self.op_foods:
            features['danger_zone'] = 1

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[successor.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {
            'num_invaders': -1000,
            'on_defense': 100,
            'invader_distance': -10,
            'stop': -100,
            'reverse': -2,
            'protect_food': 500,  # Weight for protecting own food
            'danger_zone': -500   # Weight for avoiding opponent's food
        }
        
    def getReward(self, game_state, nextState):
        return 0

    



    """
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}

    """