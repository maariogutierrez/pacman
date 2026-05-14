from stable_baselines3 import DQN
from pacman_env import PacManEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
import torch as th
import torch.nn as nn
from pathlib import Path
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class PacmanCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            # First layer: 32 filters, 3x3 kernel, stride 1 (keeps size 15x15)
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Second layer: 64 filters, 3x3 kernel, stride 1 (keeps size 15x15)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Flatten layer to prepare for the fully connected part
            nn.Flatten(),
        )

        # Calculate the size of the flattened features to link to the output layer
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations))


class LatestReplayBufferCheckpoint(CheckpointCallback):
    def _on_step(self) -> bool:
        continue_training = super()._on_step()

        if self.save_replay_buffer:
            checkpoint_dir = Path(self.save_path)
            replay_buffers = sorted(checkpoint_dir.glob(f"{self.name_prefix}_replay_buffer_*_steps.pkl"))
            if len(replay_buffers) > 1:
                for replay_buffer_path in replay_buffers[:-1]:
                    replay_buffer_path.unlink(missing_ok=True)

        return continue_training
    
env = PacManEnv()

checkpoint_callback = LatestReplayBufferCheckpoint(
  save_freq=50000,           
  save_path="./checkpoints/", 
  name_prefix="dqn_pacman",
  save_replay_buffer=True,    
)

policy_kwargs = dict(
    features_extractor_class=PacmanCNN,
    features_extractor_kwargs=dict(features_dim=256),
    normalize_images=False 
)

model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1) # Using Convolutional Neural Networks to recognise spatial patterns.

model.learn(total_timesteps=int(1000000), callback=checkpoint_callback, progress_bar=True)

model.save('dqn_pacman')
del model

model = DQN.load('dqn_pacman', env=env)
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f'Mean reward: {mean_reward}')
print(f'Standard deviation reward: {std_reward}')