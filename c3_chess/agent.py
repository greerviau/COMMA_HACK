import torch
import numpy as np
from qnet import QNet
from chess import Board
from state import State

class QAgent:
    def __init__(self):
        self.qnet = QNet()
        self.replay_buffer = []
        self.discount_rate = 0.99
        
    def move(self, state):
        actions = state.edges()
        states = []
        for action in actions:
            state.board.push(action)
            states.append(state.serialize().astype(np.float32))
            state.board.pop()

        vals = self.qnet(torch.tensor(np.array(states))).detach().numpy()[:, 0]
        action = actions[np.argmax(vals)]

        return action, np.max(vals)
