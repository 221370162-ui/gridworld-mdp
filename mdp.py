import numpy as np

ACTIONS = ["U", "D", "L", "R"]
ACTION_DELTA = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1)
}

class GridWorldMDP:
    def __init__(self, grid, terminals, obstacles, gamma=0.9):
        self.grid = grid
        self.rows, self.cols = grid
        self.terminals = terminals
        self.obstacles = obstacles
        self.gamma = gamma
        self.states = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in obstacles
        ]

    def reward(self, state):
        if state in self.terminals:
            return self.terminals[state]
        return -0.1

    def is_terminal(self, state):
        return state in self.terminals

    def next_state(self, state, action):
        if self.is_terminal(state):
            return state

        dr, dc = ACTION_DELTA[action]
        r, c = state
        nr, nc = r + dr, c + dc

        if (nr < 0 or nr >= self.rows or
            nc < 0 or nc >= self.cols or
            (nr, nc) in self.obstacles):
            return state

        return (nr, nc)

    def transitions(self, state, action):
        if self.is_terminal(state):
            return [(1.0, state)]

        probs = []
        intended = action
        perpendicular = {
            "U": ["L", "R"],
            "D": ["L", "R"],
            "L": ["U", "D"],
            "R": ["U", "D"]
        }

        probs.append((0.8, self.next_state(state, intended)))
        for a in perpendicular[action]:
            probs.append((0.1, self.next_state(state, a)))

        return probs

