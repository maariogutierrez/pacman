from stable_baselines3 import DQN
from pacman_env import PacManEnv
from pacman import PacManGame

if __name__ == '__main__':
    env = PacManEnv()
    model = DQN.load("dqn_pacman", env=env)

    action_map = {0: 'U', 1: 'D', 2: 'L', 3: 'R'}

    def dqn_action(game):
        obs = game.arena.copy()
        action, _ = model.predict(obs, deterministic=True)
        return action_map[int(action)]

    game = PacManGame()
    game.start(action_fn=dqn_action)