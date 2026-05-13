from stable_baselines3 import DQN
from pacman_env import PacManEnv
from stable_baselines3.common.evaluation import evaluate_policy

env = PacManEnv()

model = DQN("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=int(2e5), progress_bar=True)

model.save('dqn_pacman')
del model

model = DQN.load('dqn_pacman', env=env)
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f'Mean reward: {mean_reward}')
print(f'Standard deviation reward: {std_reward}')