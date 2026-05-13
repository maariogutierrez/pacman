import gymnasium as gym
import numpy as np
from typing import Optional
from pacman import PacManGame
import random

class PacManEnv(gym.Env):

    def __init__(self):
        self.pacman_game = PacManGame()
        self.observation_space = gym.spaces.Box(0, 3, (15,15), np.int8)
        self.action_space = gym.spaces.Discrete(4)
        self.max_steps = 500
        self.step_count = 0

        self._action_to_direction = {
            0: 'U',   
            1: 'D',  
            2: 'L',  
            3: 'R',   
        }

    def _get_obs(self):
        return self.pacman_game.arena
    
    def _get_info(self):
        return {
            "score": self.pacman_game.score,
            "lives": self.pacman_game.lives
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.pacman_game = PacManGame()
        self.step_count = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        self.step_count += 1

        direction = self._action_to_direction[action]

        score = self.pacman_game.score
        lives = self.pacman_game.lives 

        self.pacman_game.pacman.move(self.pacman_game, direction)
        for ghost in self.pacman_game.ghosts:
            ghost.move(self.pacman_game)
        if random.random() < 0.1:
            self.pacman_game.new_reward()
        if random.random() < 0.05:
            self.pacman_game.new_ghost()

        terminated = self.pacman_game.gameover
        truncated = self.step_count >= self.max_steps
        
        reward = 10 if self.pacman_game.score > score else -1 if len(self.pacman_game.rewards) > 0 else 0
        reward -= 50 if self.pacman_game.lives < lives else 0
        reward -= 50 if terminated else 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info




if __name__ == '__main__':
    env = PacManEnv()
    obs, info = env.reset(seed=42)

    print(f"----- Starting distribution -----\n{obs}")

    actions = [0, 1, 2, 3]
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"----- Distribution after action {action} -----\n{obs}")
        print(f"Reward: {reward}")
        print(f"More info: {info}")