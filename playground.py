from envs import get_sb3_env
from greedy import greedy_action
from time import sleep
from highway_env.vehicle.behavior import IDMVehicle

if __name__ == "__main__":

    n_agents = 3
    image_obs=False
    env = get_sb3_env(n_agents=n_agents, image_obs=image_obs, density=2)

    # Example cycle
    obs, info = env.reset()
    env.render()
    done = truncated = False

    print([v.position[0] for v in env.original_env.controlled_vehicles])

    while not (done or truncated):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        # greedy_actions = greedy_action(env)
        # obs, reward, done, truncated, info = env.step(greedy_actions)
        # print(greedy_actions, reward)
 
        print([v.position[0] for v in env.original_env.controlled_vehicles])

        sleep(0.3)
        env.render()

