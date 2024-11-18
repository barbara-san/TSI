from envs import get_sb3_env
from dqn import train_DQN
from ppo import train_PPO

import yaml
from pprint import pprint


def experiment_config(experiment_id):
    with open("experiments.yaml", "r") as file:
        experiment_dicts = yaml.safe_load(file)

    exp_dict = experiment_dicts[experiment_id]
    config_dict = {}

    config_dict["algorithm"] = exp_dict["algorithm"]
    
    config_dict["with_greedy"] = exp_dict["with_greedy"] if exp_dict["algorithm"]!="PPO" else False

    config_dict["image_obs"] = (exp_dict["observation_type"]=="image")

    config_dict["n_agents"] = exp_dict["n_agents"]

    density_vals = {"low":1, "medium":2, "high":3}
    config_dict["density"] = density_vals[exp_dict["density"]]

    config_dict["initial_headway_distance"] = exp_dict["initial_headway_distance"]

    return config_dict


if __name__ == "__main__":
    experiment_id = 5
    config_dict = experiment_config(experiment_id)
    pprint(config_dict)

    multi_agent_env = get_sb3_env(
        n_agents=config_dict["n_agents"],
        image_obs=config_dict["image_obs"],
        density=config_dict["density"],
        init_headway_distance=config_dict["initial_headway_distance"],
    )

    train_function = train_DQN if config_dict["algorithm"]=="DQN" else train_PPO
    train_function(multi_agent_env, total_timesteps=1000, exp_id=f"exp_{experiment_id}", **config_dict)


