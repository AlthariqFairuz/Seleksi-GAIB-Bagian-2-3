import numpy as np
import random

class Game:
    def __init__(self):
        """ 
        Constructor for the Game class 
        """
        self.reset()

    def reset(self):
        """
        Reset the game
        """
        self.player_pos = 2
        self.score = 0
        return self.player_pos

    def step(self, action):
        """
        Function to give the reward based on the action
        """
        if action == "left":
            self.player_pos -= 1
        elif action == "right":
            self.player_pos += 1

        if self.player_pos == 0:
            reward = -100
            self.player_pos = 2
        elif self.player_pos == 9:
            reward = 100
            self.player_pos = 2
        else:
            reward = -1

        self.score += reward
        done = self.score >= 500 or self.score <= -200

        return self.player_pos, reward, done

def choose_action(state, q_table, epsilon):
    """
    Function to choose the action based on epsilon-greedy policy
    """
    if random.uniform(0, 1) < epsilon:
        
        return random.choice(["left", "right"])
    else:
        return "left" if q_table[state, 0] > q_table[state, 1] else "right"

def update_q_learning(q_table, state, action, reward, next_state, alpha, gamma):
    """
    Update the Q-table using Q-Learning
    """
    action_index = 0 if action == "left" else 1
    next_max = np.max(q_table[next_state])
    q_table[state, action_index] += alpha * (reward + gamma * next_max - q_table[state, action_index])

def update_sarsa(q_table, state, action, reward, next_state, next_action, alpha, gamma):
    """
    Update the Q-table using SARSA
    """
    action_index = 0 if action == "left" else 1
    next_action_index = 0 if next_action == "left" else 1
    q_table[state, action_index] += alpha * (reward + gamma * q_table[next_state, next_action_index] - q_table[state, action_index])

# Parameters
alpha = 0.1 # Learning rate
gamma = 0.9 # Discount factor
epsilon = 0.1 # Exploration rate
episodes = 1000 # Number of episodes

# Initialize Q-table
q_table_q_learning = np.zeros((10, 2))
q_table_sarsa = np.zeros((10, 2))

# Q-Learning
for episode in range(episodes):
    game = Game()
    state = game.reset()
    done = False
    path = [state]  # To track the player's path

    while not done:
        action = choose_action(state, q_table_q_learning, epsilon)
        next_state, reward, done = game.step(action)
        update_q_learning(q_table_q_learning, state, action, reward, next_state, alpha, gamma)
        state = next_state
        path.append(state)  # Record the player's position

# Print the path for the last episode
print("Q-Learning Path: \n", path)

# Display final Q-tables
print("\nQ-Table for Q-Learning:\n", q_table_q_learning)

# SARSA
for episode in range(episodes):
    game = Game()
    state = game.reset()
    done = False
    action = choose_action(state, q_table_sarsa, epsilon)
    path = [state]  # To track the player's path

    while not done:
        next_state, reward, done = game.step(action)
        next_action = choose_action(next_state, q_table_sarsa, epsilon)
        update_sarsa(q_table_sarsa, state, action, reward, next_state, next_action, alpha, gamma)
        state, action = next_state, next_action
        path.append(state)  # Record the player's position

# Print the path for the last episode
print("\nSARSA Path:\n", path)

# Display final Q-tables
print("\nQ-Table for SARSA:\n", q_table_sarsa)