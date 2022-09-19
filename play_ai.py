import torch
import chess
import random
import numpy
from create_data import random_board, stockfish, split_dims
from model import NeuralNet

model = NeuralNet()
model.load_state_dict(torch.load("best_model.h5"))
model.eval()

def evaluate(board):
    position = split_dims(board)
    position = numpy.reshape(position,
    newshape = (1, 12, 8, 8))
    position = torch.Tensor(position)

    return model(position)

def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return evaluate(board)
    
    if maximizing_player:
        value = -numpy.Inf

        for move in board.legal_moves:
            board.push(move)
            value = max(value, minimax(board, depth - 1, alpha, beta, False))
            alpha = max(alpha, value)
            board.pop()
            if beta <= alpha:
                break
        
        return value

    else:
        value = numpy.Inf

        for move in board.legal_moves:
            board.push(move)
            value = min(value, minimax(board, depth - 1, alpha, beta, True))
            beta = min(beta, value)
            board.pop()
            if beta <= alpha:
                break
    
    return value
    

def get_ai_move(board, depth):
    max_move = None
    max_eval = -numpy.Inf

    for move in board.legal_moves:
        board.push(move)
        eval = minimax(board, depth - 1, -numpy.Inf, numpy.Inf, False)
        board.pop()
        if eval > max_eval:
            max_eval = eval
            max_move = move

    return max_move

board = chess.Board(chess.STARTING_FEN)

with chess.engine.SimpleEngine.popen_uci("stockfish_15_linux_x64/stockfish_15_x64") as engine:
    while True:
        move = get_ai_move(board, 1)
        board.push(move)
        print(f"\n{board}")
        if board.is_game_over():
            break
        
        move = engine.analyse(board, chess.engine.Limit(time = 1), info=chess.engine.INFO_PV)['pv'][0]
        board.push(move)
        print(f"\n{board}")
        if board.is_game_over():
            break