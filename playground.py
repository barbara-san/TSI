from envs import get_sb3_env
from greedy import greedy_action
from time import sleep

if __name__ == "__main__":

    n_agents = 3
    image_obs=False
    env = get_sb3_env(n_agents=n_agents, image_obs=image_obs)

    # Example cycle
    obs, info = env.reset()
    env.render()
    done = truncated = False
    
    while not (done or truncated):
        # action = env.action_space.sample()
        # obs, reward, done, truncated, info = env.step(action)

        greedy_actions = greedy_action(env)
        obs, reward, done, truncated, info = env.step(greedy_actions)
        print(greedy_actions, reward)
 
        print([vehicle.crashed for vehicle in env.controlled_vehicles])
        sleep(0.3)
        env.render()

