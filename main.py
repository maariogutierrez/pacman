from stable_baselines3 import DQN
from pacman_env import PacManEnv
from pacman import PacManGame
import numpy as np

if __name__ == '__main__':
    env = PacManEnv()
    model = DQN.load("checkpoints/dqn_pacman_50000_steps", env=env)

    action_map = {0: 'U', 1: 'D', 2: 'L', 3: 'R'}

    def dqn_action(game):
        arena = game.arena
        obs = np.zeros((5, 15, 15), dtype=np.float32)
        for i in range(4):
            obs[i] = (arena == i).astype(np.float32)
        obs[4] = np.clip(game.visit_counts / 10.0, 0, 1)
        action, _ = model.predict(obs, deterministic=True)
        return action_map[int(action)]

    game = PacManGame()
    game.start(action_fn=dqn_action)