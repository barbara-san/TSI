import numpy as np
from tqdm import tqdm
import pickle
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor

from envs import MultiAgentHighwayEnv, get_sb3_env

def evaluate_actions(env: MultiAgentHighwayEnv, action: int):
    obs, reward, done, truncated, info = env.step(action)
    return action, reward

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

### the version above is more efficient than a sequential/non-parallized version, but the deepcopy function is adding overhead


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
