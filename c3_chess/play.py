import chess
from state import State
from agent import QAgent

def game_over(board):
    return board.is_checkmate() or board.is_stalemate() or board.is_variant_end() or board.is_repetition()

def self_play():
    rook_endgame = chess.Board("1r2k2r/8/8/8/8/8/8/R3K1R1 w")
    qagent = QAgent()

    game_buffer = []

    while not game_over(rook_endgame):
        state = State(rook_endgame)

        move, max_val = qagent.move(state)
        rook_endgame.push(move)

        r =  0

    return game_buffer 


if __name__ == "__main__":
    game_buffer = self_play()


