from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import DQN
import torch
import torch.nn as nn
import pickle
from gymnasium import spaces

from stable_baselines3.common.logger import configure
from logger import CustomLogger

from envs import MultiAgentHighwayEnv


class ConvFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Tuple, n_agents: int, channels_per_agent: int = 1, features_dim: int = 128):
        super(ConvFeatureExtractor, self).__init__(observation_space, features_dim)

        self.n_agents = n_agents
        self.channels_per_agent = channels_per_agent
        *_, height, width = observation_space.shape
        
        # convolutional layers to process each 2D observation, each channel corresponds to an agent's observation
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.n_agents * channels_per_agent, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # compute the output dimension after convolutional layers
        conv_output_dim = self._get_conv_output_dim(height, width)

        # define a fully connected layer to map to features_dim
        self.fc = nn.Linear(conv_output_dim, features_dim) 

    def _get_conv_output_dim(self, height, width):
        # run a dummy input through the conv layers to determine output size
        with torch.no_grad():
            x = torch.zeros(1, self.n_agents*self.channels_per_agent, height, width)
            x = self.conv_layers(x)
            return x.numel()  # total number of elements after convolutions
    
    def forward(self, observations):
        # each observation is of shape (B, n_agents, H, W)
        if self.channels_per_agent > 1:
            shape = observations.shape
            observations = observations.view(-1, self.n_agents*self.channels_per_agent, shape[-2], shape[-1])
        conv_output = self.conv_layers(observations)
        return torch.relu(self.fc(conv_output))


def train_DQN(multi_agent_env: MultiAgentHighwayEnv, total_timesteps: int, exp_id: str, device: str, **exp_congig):
    multi_agent_env.reset()

    channels_per_agent = 1 if not multi_agent_env.image_obs else multi_agent_env.original_env.config["observation"]["observation_config"]["stack_size"]

    features_dim = 256
    policy_kwargs = dict(
        features_extractor_class=ConvFeatureExtractor,
        features_extractor_kwargs=dict(
            n_agents=multi_agent_env.n_agents,
            channels_per_agent=channels_per_agent,
            features_dim=features_dim
        )
    )
    algorithm_params = {
        "learning_rate": 0.005,
        "gamma": 0.99,
        "buffer_size": 10000,
        "exploration_initial_eps": 0.9,
        "exploration_final_eps": 0.15,
    }
    model = DQN(policy="MlpPolicy", env=multi_agent_env, verbose=1, policy_kwargs=policy_kwargs, device=device, **algorithm_params)

    # load pre-experience from greedy strategy
    if exp_congig["with_greedy"]:
        greedy_buffer = []
        greedy_buffer_filename = f"agents{multi_agent_env.n_agents}.pkl"
        with open(f"greedy_buffers/{greedy_buffer_filename}", "rb") as file:
            greedy_buffer = pickle.load(file)

        # add the greedy transitions to the replay buffer
        replay_buffer = model.replay_buffer
        for *transition_tuple, truncated in greedy_buffer:
            replay_buffer.add(*transition_tuple, infos=[{"TimeLimit.truncated": truncated}])
        del greedy_buffer

    # train the model
    logger = configure(f"logs/DQN/DQN_{exp_id}", ["csv", "tensorboard"])
    model.set_logger(logger)

    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=CustomLogger(model_type='DQN'))
    model.save(path=f"./models/DQN/DQN_{exp_id}")
