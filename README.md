# Pacman Agent - Markov Decision Process and Reinforcement Learning

## Project Overview

This project implements a Q-learning agent for the Pacman environment as part of the *ECE421: Introduction to Machine Learning* course. The task involves applying reinforcement learning techniques to control Pacman and optimize its pathfinding within a grid environment.

In this project, we:
- Implemented a Q-learning agent for Pacman.
- Extended the Q-learning approach with epsilon-greedy action selection.
- Tuned the agent's learning process by experimenting with various parameters and analyzing the behavior.

## Files

- **qlearningAgents.py**: Contains the main Q-learning agent implementation.
- **valueIterationAgents.py**: Implements value iteration for solving known MDPs.
- **analysis.py**: Contains answers to theoretical questions about the algorithms.
- **pacman.py**: The core program to run the Pacman simulation, using Q-learning or Approximate Q-learning agents.

## Installation

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/yourusername/pacman-agent.git
    ```
2. Navigate to the project directory:
    ```bash
    cd pacman-agent
    ```

## Setup Instructions

Ensure that you have Python 3.x installed on your system. The project is developed with Python, and you can use the following steps to set it up:

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. **Pacman Environment Setup**:
    - The Pacman environment is provided in the starter code. The simulation runs by controlling a Q-learning agent using keyboard inputs and various learning parameters.
    - You can test the agent using different environments such as smallGrid, mediumGrid, and mediumClassic.

## Usage

### Running the Q-Learning Agent

To train the Q-learning agent for Pacman, run the following command:

```bash
python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid
```
This will train the agent for 2000 episodes and test it for 10 games on a small grid.

# Running Q-Learning with Custom Settings
You can adjust the learning parameters (epsilon, alpha, gamma) for different experiments:
```bash
python pacman.py -p PacmanQAgent -a epsilon=0.1,alpha=0.5,gamma=0.9 -x 2000 -n 2010 -l smallGrid
```
# Approximate Q-Learning Agent
To experiment with Approximate Q-learning, use the following:
```bash
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid
```
# License
This project is licensed under the MIT License - see the LICENSE file for details.
