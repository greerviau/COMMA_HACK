import chess
import torch
import random
from torch import nn
from torch import optim
import numpy as np
from state import State
from agent import QAgent

def self_play(qagent, temp=0.0, render=False):
    rook_endgame = chess.Board("1r2k2r/8/8/8/8/8/8/R3K1R1 w")

    game_buffer = []
    while rook_endgame.outcome() == None:
        st = State(rook_endgame)
        game_buffer.append([st.serialize().astype(np.float32)])
        if render:
            print(st.board)
            print("==========")

        move, max_val_t1 = qagent.move(st, temp)
        rook_endgame.push(move)

    rt = rook_endgame.is_checkmate()
    if render:
        print(st.board)
        print("==========")

    for i in range(len(game_buffer)):
        game_buffer[i].append((-1.0)**((len(game_buffer) - i - 1)%2) * qagent.discount_rate**(len(game_buffer) - i - 1))
    return game_buffer, rt


def play_random_agent(qagent, ngames=1, render=False):
    for game in range(ngames):
        rook_endgame = chess.Board("1r2k2r/8/8/8/8/8/8/R3K1R1 w")
        qagent_turn = game % 2

        while rook_endgame.outcome() == None:
            if render:
                print(rook_endgame)
        
            if qagent_turn == rook_endgame.turn:
                st = State(rook_endgame)
                move, _ = qagent.move(st, 0.0)
            else:
                move = random.choice(list(rook_endgame.legal_moves))
            rook_endgame.push(move)

        if render:
            print(rook_endgame)

        if rook_endgame.outcome().winner == None:
            print("Draw")
        elif qagent_turn == rook_endgame.outcome().winner:
            print("Won")
        else:
            print("Lost")

        
if __name__ == "__main__":
    qagent = QAgent()
    qagent.qnet = torch.load("qnet.pth")
    game_count = 1000
    epochs = 10

    mse_loss = nn.MSELoss()
    opt = optim.Adam(qagent.qnet.parameters(), lr=3e-4)

    play_random_agent(qagent, 100, False)
    quit()

    replay_buffer = []

    for game in range(game_count):
        print(f"Game {game}")
        game_buffer, cm = self_play(qagent, temp=0.5)
        if cm:
            print("cm")
            replay_buffer += game_buffer

    for e in range(epochs):
        tens = torch.tensor(np.array([s[0] for s in replay_buffer]))
        y = torch.tensor(np.array([s[1] for s in replay_buffer])[..., None].astype(np.float32))
        pred = qagent.qnet(tens)
        loss = mse_loss(pred, y)
        print(loss)

        opt.zero_grad()
        loss.backward()
        opt.step()

    # play_random_agent(qagent, 1, True)
    torch.save(qagent.qnet, "qnet.pth")
