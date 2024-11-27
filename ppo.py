from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
import torch
import torch.nn as nn
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


def train_PPO(multi_agent_env: MultiAgentHighwayEnv, total_timesteps: int, exp_id: str, device: str, **exp_congig):
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
        "batch_size": 32,
        "n_steps": 512
    }

    # train the model
    logger = configure(f"logs/PPO/PPO_{exp_id}", ["csv", "tensorboard"])

    model = PPO(policy="MlpPolicy", env=multi_agent_env, verbose=1, policy_kwargs=policy_kwargs, device=device, **algorithm_params)
    model.set_logger(logger)
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=CustomLogger(model_type='PPO'))
    model.save(path=f"./models/PPO/PPO_{exp_id}")

