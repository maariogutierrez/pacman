import gymnasium as gym
import numpy as np
from typing import Optional
from pacman import PacManGame
import random

class PacManEnv(gym.Env):

    def __init__(self):
        self.pacman_game = PacManGame()
        self.observation_space = gym.spaces.Box(0, 1, (4,15,15), np.int8)
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
        arena = self.pacman_game.arena
        # Create 4 binary planes: 0=Empty, 1=PacMan, 2=Ghost, 3=Pellet
        obs = np.zeros((4, 15, 15), dtype=np.int8)
    
        for i in range(4):
            obs[i] = (arena == i).astype(np.int8)
        return obs
    
    def _get_distance_to_nearest_pellet(self):
        """Calculates Manhattan distance in a toroidal (wrapping) grid."""
        pacman_pos = self.pacman_game.pacman.position
        pellet_indices = np.argwhere(self.pacman_game.arena == 3)
        if len(pellet_indices) == 0:
            return 0
        
        # Calculate distance considering the 15x15 wrap-around
        diff = np.abs(pellet_indices - pacman_pos)
        wrapped_diff = np.minimum(diff, 15 - diff)
        distances = wrapped_diff.sum(axis=1)
        return np.min(distances)
    
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
        before_distance = self._get_distance_to_nearest_pellet()

        self.pacman_game.pacman.move(self.pacman_game, direction)
        for ghost in self.pacman_game.ghosts:
            ghost.move(self.pacman_game)
        if random.random() < 0.1:
            self.pacman_game.new_reward()
        if random.random() < 0.05:
            self.pacman_game.new_ghost()

        after_distance = self._get_distance_to_nearest_pellet()
        terminated = self.pacman_game.gameover
        truncated = self.step_count >= self.max_steps
        
        reward = -0.1

        reward += 50 if self.pacman_game.score > score else 0
        reward -= 100 if self.pacman_game.lives < lives else 0
        reward -= 100 if terminated else 0

        if before_distance > 0 and after_distance > 0:
            if after_distance < before_distance:
                reward += 1.0  
            elif after_distance > before_distance:
                reward -= 1.2

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