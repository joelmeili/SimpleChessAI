import chess
import chess.engine
import random
import numpy

def random_board(max_depth = 200):
    board = chess.Board()
    depth = random.randrange(0, max_depth)

    for _ in range(depth):
        all_moves = list(board.legal_moves)
        random_move = random.choice(all_moves)
        board.push(random_move)

        if board.is_game_over():
            break

    return board

def stockfish(board, depth):
        with chess.engine.SimpleEngine.popen_uci("stockfish_15_linux_x64/stockfish_15_x64") as sf:
            result = sf.analyse(board, chess.engine.Limit(depth = depth))
            score = result["score"].white().score()
        
        return score

def split_dims(board):
    board3d = numpy.zeros((12, 8, 8), dtype = numpy.int8)
    
    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            idx = numpy.unravel_index(square, (8, 8))
            board3d[piece - 1][7 - idx[0]][idx[1]] = 1

        for square in board.pieces(piece, chess.BLACK):
            idx = numpy.unravel_index(square, (8, 8))
            board3d[piece + 5][7 - idx[0]][idx[1]] = 1
            
    return board3d

if __name__ == "create_data":

    random.seed(2022)

    positions = []
    scores = []

    iter = 10000
    count = 0

    while (count < iter):
        board = random_board()
        position = split_dims(board)
        score = stockfish(board, depth = 5)

        if score is not None:
            positions.append(position)
            scores.append(score)
            count += 1

    scores = numpy.asarray(scores, dtype=numpy.int32)

    numpy.savez("data.npz", positions, scores)