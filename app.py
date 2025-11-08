import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
import sys
# Configure stdout encoding
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Input
import random
import pygame
import time
# Maze setup (1 = valid path, 0 = wall)
maze = [
    [1, 0, 1, 1],
    [1, 0, 1, 0],
    [1, 1, 1, 1],
    [0, 0, 0, 1]
]

ROWS, COLS = len(maze), len(maze[0])
START = (0, 0)  # Starting point
GOAL = (ROWS-1, COLS-1)  # Goal point

# Function to generate the current state
def get_state(maze, pos):
    state = np.array(maze)
    if pos[0] < ROWS and pos[1] < COLS and pos[0] >= 0 and pos[1] >= 0:
        state[pos] = 3  # Mark the player's position
    return state

# Function to determine possible actions
def get_possible_actions(maze, pos):
    actions = []
    x, y = pos
    if x > 0 and maze[x-1][y] != 0:  # up
        actions.append(0)
    if x < ROWS - 1 and maze[x+1][y] != 0:  # down
        actions.append(1)
    if y > 0 and maze[x][y-1] != 0:  # left
        actions.append(2)
    if y < COLS - 1 and maze[x][y+1] != 0:  # right
        actions.append(3)
    return actions

# Function to move the player
def move_player(pos, action):
    x, y = pos
    if action == 0:
        return (x-1, y)
    elif action == 1:
        return (x+1, y)
    elif action == 2:
        return (x, y-1)
    elif action == 3:
        return (x, y+1)
    return pos

# Neural network model setup
model = Sequential([
  Input(shape=(ROWS, COLS)),
  Flatten(),
  Dense(32, activation='relu'),
  Dense(32, activation='relu'),
  Dense(4, activation='linear')
])

model.compile(optimizer='adam', loss='mse')

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((400, 400))
pygame.display.set_caption('Q-Learning Maze')
clock = pygame.time.Clock()

# Function to draw the maze
def draw_maze(maze, pos):
    screen.fill((255, 255, 255))
    for i in range(ROWS):
        for j in range(COLS):
            color = (255, 255, 255)
            if maze[i][j] == 0:
                color = (0, 0, 0)
            pygame.draw.rect(screen, color, pygame.Rect(j * 100, i * 100, 100, 100))
    pygame.draw.circle(screen, (0, 0, 255), (pos[1] * 100 + 50, pos[0] * 100 + 50), 40)
    pygame.display.flip()

# Train the model using Q-Learning with visualization
def train_model(epochs, gamma=0.9, epsilon=0.1, alpha=0.01):

    state = get_state(maze, START)
    q_values = model.predict(state.reshape(1, ROWS, COLS))
    for epoch in range(epochs):
        state = get_state(maze, START)
        pos = START
        total_reward = 0
        steps = 0
        # Display the maze and agent at each step during training
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        draw_maze(maze, pos)
        clock.tick(2)  # Control update speed
       
        while pos != GOAL:
            possible_actions = get_possible_actions(maze, pos)
            if not possible_actions:
                break  # If no possible moves, exit loop
            if random.uniform(0, 1) < epsilon:
                action = random.choice(possible_actions)
            else:
                q_values = model.predict(state.reshape(1, ROWS, COLS))
                action = possible_actions[np.argmax(q_values[0][possible_actions])]

            new_pos = move_player(pos, action)
            if new_pos[0] >= ROWS or new_pos[1] >= COLS or new_pos[0] < 0 or new_pos[1] < 0:
                reward = -1  # Penalize moves outside boundaries
                new_pos = pos  # Stay in current position
            else:
                reward = 1 if new_pos == GOAL else -0.1
                new_state = get_state(maze, new_pos)
                q_update = reward + gamma * np.max(model.predict(new_state.reshape(1, ROWS, COLS)))
                q_values[0, action] = q_values[0, action] + alpha * (q_update - q_values[0, action])
                model.fit(state.reshape(1, ROWS, COLS), q_values, epochs=1, verbose=0)

                state = new_state
                pos = new_pos
                total_reward += reward
                steps += 1
            
            # Display maze and agent at each step during training
            pygame.display.set_caption(f"Epoch {epoch} :Epsilon = {epsilon:.2f},Reward = {total_reward:.2f} ")

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            draw_maze(maze, pos)
            clock.tick(2)  # Control update speed
        
        print(f"Epoch {epoch}: Total Reward = {total_reward}")

# Function to test the trained model
def test_model():
    state = get_state(maze, START)
    pos = START
    steps = 0
    while pos != GOAL and steps < 100:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        draw_maze(maze, pos)  # Draw maze at each step
        time.sleep(1)  # Add 1 second delay between steps
        q_values = model.predict(state.reshape(1, ROWS, COLS))
        possible_actions = get_possible_actions(maze, pos)
        action = possible_actions[np.argmax(q_values[0][possible_actions])]

        new_pos = move_player(pos, action)
        # Check that the new move is within bounds and not a wall
        if new_pos[0] >= ROWS or new_pos[1] >= COLS or new_pos[0] < 0 or new_pos[1] < 0 or maze[new_pos[0]][new_pos[1]] == 0:
            print(f"Invalid move: Position = {new_pos}")
            new_pos = pos
        else:
            pos = new_pos
        state = get_state(maze, pos)
        steps += 1
        print(f"Step {steps}: Position = {pos}")
        clock.tick(1)  # Control update speed

# Train the model
train_model(50)
model.save('model.keras')
if os.path.exists('model.keras'):
    tf.keras.models.load_model('model.keras')
else:
    print("The model file does not exist.")

pygame.display.set_caption('Q-Learning Maze-test')
# Test the trained model
test_model()
