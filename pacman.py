import numpy as np
import time
import random
import os
from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.align import Align
from rich import box
from rich.live import Live


console = Console()

CELL = {
    0: Text('·', style='white'),
    3: Text('●', style='bold cyan'),
    1: Text('ᗤ', style='bold yellow'),
    2: Text('ᗣ', style='bold red'),
}

PACMAN_CHARS = {
    'R': 'ᗤ',
    'L': 'ᗧ',
    'U': 'ᗢ',
    'D': 'ᗣ',
}


class PacMan:
    def __init__(self):
        self.name = "PacMan"
        self.position = np.array([7, 7])
        self.direction = 'R'

    def move(self, game, direction='R'):
        self.direction = direction
        game.arena[self.position[0], self.position[1]] = 0
        if not game.gameover:
            if game.respawn:
                self.respawn(game)
            else:
                self.position = (self.position + game.moves[direction]) % 15
                if game.arena[self.position[0], self.position[1]] == 3:
                    game.score += 1
                    for i, reward in enumerate(game.rewards):
                        if np.array_equal(reward.position, self.position):
                            game.rewards.pop(i)
                            break
                    game.arena[self.position[0], self.position[1]] = 1
                elif game.arena[self.position[0], self.position[1]] == 2:
                    game.lives -= 1
                    if game.lives == 0:
                        game.gameover = True
                    else:
                        self.respawn(game)
                else:
                    game.arena[self.position[0], self.position[1]] = 1

    def respawn(self, game):
        candidates = np.array([[0, 0], [0, 14], [14, 0], [14, 14]])
        distances = np.linalg.norm(candidates[:, np.newaxis] - self.position, axis=2)
        self.position = candidates[np.argmax(distances.sum(axis=1))]
        game.respawn = False
        game.arena[self.position[0], self.position[1]] = 1


class Reward:
    def __init__(self):
        self.status = 'Inactive'

    def activate(self, pacman):
        while True:
            position = np.array([random.randint(0, 14), random.randint(0, 14)])
            if np.all(position != pacman.position):
                self.position = position
                self.status = 'Active'
                break


class Ghost:
    def __init__(self):
        self.status = 'Inactive'

    def activate(self, pacman):
        while True:
            position = np.array([random.randint(0, 14), random.randint(0, 14)])
            if np.all(position != pacman.position):
                self.position = position
                self.status = 'Active'
                break

    def move(self, game):
        game.arena[self.position[0], self.position[1]] = (3 if any(np.array_equal(self.position, reward.position) for reward in game.rewards) else 0)

        xd = game.pacman.position[1] - self.position[1]
        yd = game.pacman.position[0] - self.position[0]

        if abs(xd) > abs(yd):
            self.position += game.moves['R' if xd > 0 else 'L']
        else:
            self.position += game.moves['D' if yd > 0 else 'U']

        cell = game.arena[self.position[0], self.position[1]]
        if cell == 1:
            game.respawn = True
            game.lives -= 1
            if game.lives == 0:
                game.gameover = True

        game.arena[self.position[0], self.position[1]] = 2


class PacManGame:
    def __init__(self):
        self.arena = np.full((15, 15), 0, dtype=np.int8)
        self.pacman = PacMan()
        self.arena[7, 7] = 1
        self.moves = {
            'U': np.array([-1, 0]),
            'D': np.array([1, 0]),
            'L': np.array([0, -1]),
            'R': np.array([0, 1]),
        }
        self.score = 0
        self.lives = 3
        self.rewards = []
        self.ghosts = []
        self.respawn = False
        self.gameover = False
        self.visit_counts = np.zeros((15, 15), dtype=np.float32)
        self.visit_decay = 0.99

    def new_reward(self):
        if len(self.rewards) < 5:
            r = Reward()
            r.activate(self.pacman)
            self.arena[r.position[0], r.position[1]] = 3
            self.rewards.append(r)

    def new_ghost(self):
        if len(self.ghosts) < 3:
            g = Ghost()
            g.activate(self.pacman)
            self.arena[g.position[0], g.position[1]] = 2
            self.ghosts.append(g)

    def _render_arena(self):
        table = Table(
            box=box.SIMPLE,
            show_header=False,
            padding=(0, 0),
            collapse_padding=True,
        )
        for _ in range(15):
            table.add_column(justify='center', min_width=2, no_wrap=True)

        for row in range(15):
            cells = []
            for col in range(15):
                val = self.arena[row, col]
                if val == 1:
                    ch = PACMAN_CHARS.get(self.pacman.direction, 'ᗤ')
                    cells.append(Text(f'{ch} ', style='bold yellow'))
                elif val == 2:
                    cells.append(Text('ᗣ ', style='bold red'))
                elif val == 3:
                    cells.append(Text('● ', style='bold cyan'))
                else:
                    cells.append(Text('· ', style='dim #444444'))
            table.add_row(*cells)

        return table

    def _render_hud(self):
        lives_display = Text()
        lives_display.append('♥ ' * self.lives, style='bold red')
        lives_display.append('♡ ' * (3 - self.lives), style='dim red')

        score_text = Text()
        score_text.append('SCORE  ', style='dim white')
        score_text.append(str(self.score), style='bold yellow')

        lives_text = Text()
        lives_text.append('LIVES  ', style='dim white')
        lives_text.append(lives_display)

        ghosts_text = Text()
        ghosts_text.append('GHOSTS  ', style='dim white')
        ghosts_text.append(str(len(self.ghosts)), style='bold red')

        pellets_text = Text()
        pellets_text.append('PELLETS  ', style='dim white')
        pellets_text.append(str(len(self.rewards)), style='bold cyan')

        return Columns(
            [score_text, lives_text, ghosts_text, pellets_text],
            equal=True,
            expand=True,
        )

    def _print_frame(self):
        title = Text('P A C - M A N', style='bold yellow', justify='center')

        arena_table = self._render_arena()
        hud = self._render_hud()

        return Panel(
            Group(
                Align.center(arena_table),
                hud,
            ),
            title=title,
            border_style='bright_yellow',
            padding=(0, 2),
            box=box.DOUBLE_EDGE,
        )

    def _print_gameover(self):
        os.system('clear' if os.name == 'posix' else 'cls')

        msg = Text(justify='center')
        msg.append('\n  G A M E   O V E R  \n\n', style='bold red')
        msg.append(f'  Final Score: ', style='dim white')
        msg.append(f'{self.score}\n', style='bold yellow')

        panel = Panel(
            Align.center(msg),
            border_style='red',
            box=box.DOUBLE_EDGE,
            padding=(1, 4),
        )
        console.print()
        console.print(Align.center(panel))
        console.print()
        input('Press Enter to exit...')

    def start(self, action_fn=None):
        with Live(console=console, refresh_per_second=10, screen=True) as live:
            while not self.gameover:
                self.visit_counts *= self.visit_decay  
                pos = self.pacman.position
                self.visit_counts[pos[0], pos[1]] += 1

                if action_fn is None:
                    direction = random.choice(list(self.moves.keys()))
                else:
                    direction = action_fn(self)

                self.pacman.move(self, direction)
                for ghost in self.ghosts:
                    ghost.move(self)
                if random.random() < 0.1:
                    self.new_reward()
                if random.random() < 0.05:
                    self.new_ghost()

                content = self._print_frame()
                time.sleep(0.4)
                live.update(content)

        self._print_gameover()
    