"""
This file purpose is to implement a greedy algorithm, so that given an environment on any configuration, it yields the action that returns the highest enviroment-defined reward.
Furthermore, this file also implements a function to create and save a greedy experience buffer, which is needed to train the models in the initial steps of some experiences.
"""

import numpy as np
from tqdm import tqdm
import pickle
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor

from envs import MultiAgentHighwayEnv, get_sb3_env

# necessary for multiprocessing
def evaluate_actions(env: MultiAgentHighwayEnv, action: int):
    obs, reward, done, truncated, info = env.step(action)
    return action, reward

# function that given an environment, it returns the action which yields the highest environment-defined reward when applied
# it runs in a distributed manner using ProcessPoolExecutor, being mostly faster than a sequential version of the same function
def greedy_action(env: MultiAgentHighwayEnv):
    max_reward = -np.inf
    max_actions = -1

    # parallelize action evaluation
    with ProcessPoolExecutor() as executor:
        # deepcopy of the environment for each action to avoid interference
        futures = [executor.submit(evaluate_actions, deepcopy(env), actions) for actions in range(env.action_space.n)]

        for future in futures:
            actions, reward = future.result()
            if reward > max_reward:
                max_reward = reward
                max_actions = actions
    
    executor.shutdown()

    return np.asarray([max_actions])

# function that creates a greedy experience buffer and saves it, given the desire size for the buffer and the number of agents in the environment
def create_greedy_buffer(buffer_size, n_agents):
    buffer = []

    env = get_sb3_env(n_agents=n_agents, image_obs=False, density=1)
    obs, _ = env.reset()
    done = False
  
    pbar = tqdm(total=buffer_size, desc="Greedy buffer")
    steps = 0
    while steps < buffer_size:
        while not done:
            action = greedy_action(env)
            next_obs, reward, done, truncated, _ = env.step(action)
            terminated = done or truncated
            buffer.append((obs, next_obs, action, reward, terminated, truncated))
            obs = next_obs
            
            pbar.update(1)
            steps += 1
            if steps >= buffer_size:
                break

        obs, _ = env.reset()
        done = False

    with open(f"greedy_buffers/agents{n_agents}.pkl", "wb") as file:
        pickle.dump(buffer, file)
        file.close()
    del buffer


if __name__ == "__main__":
    buffer_size = 2500
    n_agents = 3
    create_greedy_buffer(buffer_size, n_agents)
