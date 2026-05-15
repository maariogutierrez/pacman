import gymnasium as gym
import numpy as np
from typing import Optional
from pacman import PacManGame
import random
from collections import deque

class PacManEnv(gym.Env):

    def __init__(self):
        self.pacman_game = PacManGame()
        self.observation_space = gym.spaces.Box(0, 1, (5,15,15), np.float32)
        self.action_space = gym.spaces.Discrete(4)
        self.max_steps = 500
        self.step_count = 0
        self.visit_counts = np.zeros((15, 15), dtype=np.float32)
        self.visit_decay = 0.99 

        self._action_to_direction = {
            0: 'U',   
            1: 'D',  
            2: 'L',  
            3: 'R',   
        }

    def _get_obs(self):
        arena = self.pacman_game.arena
        # Create 5 binary planes: 0=Empty, 1=PacMan, 2=Ghost, 3=Pellet, 4=Visits heatmap
        obs = np.zeros((5, 15, 15), dtype=np.float32)
    
        for i in range(4):
            obs[i] = (arena == i).astype(np.float32)

        obs[4] = np.clip(self.visit_counts / 10.0, 0, 1)
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
        self.position_history = deque(maxlen=6)
        self.visit_counts = np.zeros((15, 15), dtype=np.float32)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        self.step_count += 1

        self.visit_counts *= self.visit_decay  
        pos = self.pacman_game.pacman.position
        self.visit_counts[pos[0], pos[1]] += 1

        direction = self._action_to_direction[action]

        score = self.pacman_game.score
        lives = self.pacman_game.lives 
        before_distance = self._get_distance_to_nearest_pellet()

        self.pacman_game.pacman.move(self.pacman_game, direction)
        for ghost in self.pacman_game.ghosts:
            ghost.move(self.pacman_game)

        after_distance = self._get_distance_to_nearest_pellet()

        if random.random() < 0.1:
            self.pacman_game.new_reward()
        if random.random() < 0.05:
            self.pacman_game.new_ghost()

        terminated = self.pacman_game.gameover
        truncated = self.step_count >= self.max_steps
        
        reward = -0.01 # time penalty
        
        pos = tuple(self.pacman_game.pacman.position)
        if pos in self.position_history:
            reward -= 0.5 # loop avoidance penalty 
        self.position_history.append(pos)

        pellet_eaten = self.pacman_game.score > score
        reward += 1.0 if pellet_eaten else 0 # pellet eaten reward

        if self.pacman_game.lives < lives:
            reward -= 2.0 # lost life penalty

        if not pellet_eaten and before_distance > 0 and after_distance > 0:
            if after_distance < before_distance:
                reward += 0.05  # go to pellet reward
            elif after_distance > before_distance:
                reward -= 0.06 # don't go to pellet penalty

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