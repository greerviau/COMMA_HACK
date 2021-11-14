import torch
import random
import numpy as np
from qnet import QNet
from chess import Board
from state import State

class QAgent:
    def __init__(self):
        self.qnet = QNet()
        self.replay_buffer = []
        self.discount_rate = 0.9
        
    def move(self, state, temp):
        actions = state.edges()
        states = []
        for action in actions:
            state.board.push(action)
            states.append(state.serialize().astype(np.float32))
            state.board.pop()

        vals = self.qnet(torch.tensor(np.array(states))).detach().numpy()[:, 0]
        max_val = np.max(vals)
        action = actions[np.argmax(vals)]

        if random.random() > temp:
            ind = random.randint(0, len(actions) - 1)
            action = actions[ind]
            max_val = vals[ind]

        return action, max_val
